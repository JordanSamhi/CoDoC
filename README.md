# CoBaClaSS
**Co**de-**Ba**sed **Cla**ssification of **S**ources and **S**inks.

In this repository, one can find relevant information about our approach to automatically classify Android sources and sinks.
All the artefacts necessary to reproduce our study can be found here.

## Getting started

### Downloading the tool

<pre>
git clone https://github.com/JordanSamhi/CoBaClaSS.git
</pre>

### Installing the tool

<pre>
cd CoBaClaSS
mvn clean install
</pre>

### Using the tool

#### Extracting source code and documentation from android sources:

<pre>
java -jar CoBaClaSS/target/CoBaClaSS-0.0.1-SNAPSHOT-jar-with-dependencies.jar <i>options</i>
</pre>

Options:

* ```-c``` : Extract source code.
* ```-d``` : Extract documentation.
* ```-o``` : Destination folder for files.
* ```-s``` : Source folder.

Resulting files will be stored in path given with ```-o``` option in two separate folders **source_code** and **documentation**.

#### Generating documentation vectors using sentenceBERT:

<pre>
python3 vectorize_documentation_using_sentence_bert.py EXTRACTED_SOURCE_FOLDER
</pre>

Resulting vectors will be stored in current directory in file named documentation_vectors.txt.

## Built With

* [Maven](https://maven.apache.org/) - Dependency Management

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE](LICENSE) file for details

## Contact

For any question regarding this study, please contact us at:
[Jordan Samhi](mailto:jordan.samhi@uni.lu)
