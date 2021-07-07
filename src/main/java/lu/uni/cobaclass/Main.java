package lu.uni.cobaclass;

import java.io.File;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import lu.uni.cobaclass.utils.CommandLineOptions;
import lu.uni.cobaclass.utils.Constants;
import lu.uni.cobaclass.utils.Utils;

/*-
 * #%L
 * CoBaClaSS
 * 
 * %%
 * Copyright (C) 2021 Jordan Samhi
 * University of Luxembourg - Interdisciplinary Centre for
 * Security Reliability and Trust (SnT) - TruX - All rights reserved
 *
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU GENERAL PUBLIC LICENSE Version 3 as
 * published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU GENERAL PUBLIC LICENSE for more details.
 * 
 * You should have received a copy of the GNU GENERAL PUBLIC LICENSE
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */

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
