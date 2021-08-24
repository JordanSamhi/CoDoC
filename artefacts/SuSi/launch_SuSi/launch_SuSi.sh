while getopts f: option
do
    case "${option}"
        in
        f) PSCOUT_FILE_PATH=${OPTARG};;
    esac
done

if [ -z "$PSCOUT_FILE_PATH" ]
then
    echo
    read -p "Pscout file path: " PSCOUT_FILE_PATH
fi

java -cp lib/slf4j-api-1.7.9.jar:lib/slf4j-simple-1.7.9.jar:lib/slf4j-ext-1.7.9.jar:lib/slf4j-log4j12-1.7.9.jar:lib/weka.jar:lib/soot-trunk.jar:lib/soot-infoflow.jar:lib/soot-infoflow-android.jar:SuSi.jar de.ecspride.sourcesinkfinder.SourceSinkFinder android-30.jar $PSCOUT_FILE_PATH out.pscout
