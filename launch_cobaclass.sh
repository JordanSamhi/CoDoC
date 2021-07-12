#!/bin/bash

. lib/lib.sh --source-only

if test -f "target/CoBaClaSS-0.0.1-SNAPSHOT.jaer"; then echo 1;fi

while getopts s:o: option
do
    case "${option}"
        in
        s) SOURCE_CODE_PATH=${OPTARG};;
        o) OUTPUT_PATH=${OPTARG};;
    esac
done

echo "#===========================#"
echo "|     CoBaClaSS launcher    |"
echo "#===========================#"


if [ -z "$SOURCE_CODE_PATH" ]
then
    echo
    read -p "Path of source code to handle: " SOURCE_CODE_PATH
fi
if [ -z "$OUTPUT_PATH" ]
then
    read -sp "Path of the output folder: " OUTPUT_PATH
fi

if [[ $OUTPUT_PATH == /* ]]
then
    FULL_OUTPUT_PATH=$OUTPUT_PATH
else
    FULL_OUTPUT_PATH=$(pwd)/$OUTPUT_PATH
fi

if [ ! -f target/CoBaClaSS-0.0.1-SNAPSHOT.jar ]
then
    end_program "Build CoBaClaSS first ! (./build_cobaclass.sh)"
fi

OUTPUT_PATH_DOCUMENTATION=$FULL_OUTPUT_PATH/documentation/
OUTPUT_PATH_SOURCE=$FULL_OUTPUT_PATH/source_code/

print_info "Extracting methods source code and documentation..."
cp -r $SOURCE_CODE_PATH src/main/resources/
java -jar target/CoBaClaSS-0.0.1-SNAPSHOT-jar-with-dependencies.jar -s $SOURCE_CODE_PATH -o $FULL_OUTPUT_PATH -c -d
check_return $? "Something went wrong while executing CoBaClaSS." "Source code and documentation extracted successfully in $FULL_OUTPUT_PATH."
rm -rf src/main/resources/$(basename $SOURCE_CODE_PATH)

print_info "Generating vectors from documentation"
python3 vectorize_documentation_using_sentence_bert.py $OUTPUT_PATH_DOCUMENTATION $FULL_OUTPUT_PATH
check_return $? "Something went wrong while generating vectors." "Documentation vectors generated successfully in $FULL_OUTPUT_PATH."

CODE2VEC_ANDROID_SOURCE_PATH=artefacts/code2vec/android_source/
print_info "Preparing data..."
if [ -d $CODE2VEC_ANDROID_SOURCE_PATH ]
then
    rm -rf $CODE2VEC_ANDROID_SOURCE_PATH
fi
cp -r $OUTPUT_PATH_SOURCE $CODE2VEC_ANDROID_SOURCE_PATH
print_info "Sources copied."
cd artefacts/code2vec/JavaExtractor/JPredict/
print_info "Building JavaExtractor..."
mvn clean install
print_info "JavaExtractor built."
print_info "Structuring folders..."
cd ../../android_source/
mkdir android_source_train android_source_val android_source_test
count_src=$(ls -1|grep -v android|wc -l)
five_percent=$((count_src * 5 / 100))
ls -1 |grep -v "android"|head -$five_percent|parallel -j24 mv {} android_source_val/{}.java
ls -1 |grep -v "android"|head -$five_percent|parallel -j24 mv {} android_source_test/{}.java
ls -1 |grep -v "android"|parallel -j24 mv {} android_source_train/{}.java
print_info "Data prepared."
cd ../
print_info "Preprocessing data..."
sh preprocess.sh
print_info "Data preprocessed."
print_info "Training model with code2vec with your data..."
sh train.sh
print_info "Model trained and savec in artefacts/code2vec/models/android_source/saved_model"
print_info "Generating vectors for source code with code2vec..."
python3 code2vector_infer.py -o $FULL_OUTPUT_PATH
check_return $? "Something went wrong with generating vectors." "Source code vectors successfully generated in $FULL_OUTPUT_PATH."
