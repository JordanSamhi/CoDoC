package lu.uni.cobaclass;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import lu.uni.cobaclass.utils.CommandLineOptions;
import lu.uni.cobaclass.utils.Constants;
import lu.uni.cobaclass.utils.Utils;

public class Main {
	public static void main(String[] args) throws Throwable {
		System.out.println(String.format("%s v1.0 started on %s\n", Constants.TOOL_NAME, new Date()));
		CommandLineOptions.v().parseArgs(args);
		String source_folder = CommandLineOptions.v().getSource();
		String[] split = source_folder.split("/");
		Constants.CURRENT_SOURCE_CODE_FOLDER = split[split.length - 1];
		
		SourceCodeExtractor sce = new SourceCodeExtractor(source_folder);
		File file = new File(source_folder);
		List<File> javaFiles = new ArrayList<File>();
		Utils.getJavaFilesPaths(file, javaFiles);
		String filePath = null;
		for(File f: javaFiles) {
			if(f.getPath().contains("android/")) {
				filePath = f.getAbsolutePath();
				sce.parseClass(filePath);
				sce.extractAllMethodsSourceCode();
			}
		}
		sce.writeSha256ToSootMethodNameFile();
	}
}
