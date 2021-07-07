package lu.uni.cobaclass;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.codec.digest.DigestUtils;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.GenericVisitorAdapter;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.CodeGenerationUtils;
import com.github.javaparser.utils.SourceRoot;

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
 * GNU GENERAL PUBLIC LICENSEfor more details.
 * 
 * You should have received a copy of the GNU GENERAL PUBLIC LICENSE
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/gpl-3.0.html>.
 * #L%
 */

public class SourceCodeExtractor {

	private String pathToSourceRootFolder;
	private SourceRoot sourceRoot;
	private CompilationUnit cu;
	private Map<String, String> methodToSourceCode;
	private String currentClass;
	private CombinedTypeSolver combinedTypeSolver;
	private Map<String, String> sha256ToSootMethodName;
	private String destination;

	public SourceCodeExtractor(String pathToSourceRootFolder) {
		this.pathToSourceRootFolder = pathToSourceRootFolder;
		this.sourceRoot = new SourceRoot(CodeGenerationUtils.mavenModuleRoot(SourceCodeExtractor.class).resolve(this.pathToSourceRootFolder));
		this.methodToSourceCode = new HashMap<String, String>();
		combinedTypeSolver = new CombinedTypeSolver();
		combinedTypeSolver.add(new ReflectionTypeSolver());
		combinedTypeSolver.add(new JavaParserTypeSolver(new File(String.format("%s/src/main/resources/%s", System.getProperty("user.dir"), Constants.CURRENT_SOURCE_CODE_FOLDER))));
		sha256ToSootMethodName = new HashMap<String, String>();
	}

	public void parseClass(String className) {
		String clazz = className;
		if(!className.contains(".java")) {
			clazz = String.format("%s.java", className);
		}
		this.currentClass = clazz;
		this.cu = this.sourceRoot.parse("", className);
	}

	public String extractMethodSourceCode(final String methodName) {
		if(this.cu == null) {
			throw new ExceptionInInitializerError("Compilation Unit not initialized, please run parseClass before.");
		}
		String sourceCode = cu.accept(new GenericVisitorAdapter<String, Void>() {
			public String visit(MethodDeclaration md, Void arg) {
				String sourceCode = null;
				if(Utils.removeLessAndGreaterSigns(md.getDeclarationAsString(false, false, false)).equals(methodName)) {
					sourceCode = md.toString();
				}
				return sourceCode;
			}
		}, null);
		return sourceCode;
	}

	public void extractAllMethodsSourceCode() {
		if(this.cu == null) {
			throw new ExceptionInInitializerError("Compilation Unit not initialized, please run parseClass before.");
		}
		cu.accept(new VoidVisitorAdapter<Void>() {
			public void visit(MethodDeclaration md, Void arg) {
				if(isPublic(md) && md.getComment().isPresent() && md.getBody().isPresent()) {
					String sootClassName = Utils.toSootClassName(Utils.fullPathToClassName(currentClass), md, combinedTypeSolver);
					String sha256SootClassName = DigestUtils.sha256Hex(sootClassName);
					sha256ToSootMethodName.put(sha256SootClassName, sootClassName);
					StringBuilder source_code = new StringBuilder();
					source_code.append(md.getDeclarationAsString(true, true, true));
					source_code.append(md.getBody().get());
					methodToSourceCode.put(sootClassName, source_code.toString());
					if(CommandLineOptions.v().hasOutput()) {
						destination = CommandLineOptions.v().getOutput();
					}else {
						destination = String.format("%s/cobaclass_output/", System.getProperty("user.dir"));
					}
					if(CommandLineOptions.v().hasSourceCode()) {
						writeInFile(destination, "source_code", sha256SootClassName, source_code.toString());
					}
					if(CommandLineOptions.v().hasDocumentation()) {
						writeInFile(destination, "documentation", sha256SootClassName, md.getComment().get().getContent());
					}
				}
			}
		}, null);
	}

	private void writeInFile(String destination, String folder, String filename, String content) {
		File path = new File(String.format("%s/%s", destination, folder));
		File f = new File(String.format("%s/%s", path.getPath(), filename));
		if(!path.exists()) {
			path.mkdirs();
		}
		try {
			if (f.createNewFile()) {
				System.out.println(String.format("Writing file %s...", filename));
				FileWriter fw = new FileWriter(f);
				fw.write(content);
				fw.close();
				System.out.println("Done.");
			} else {
				System.err.println(String.format("File %s already exists.", filename));
			}
		} catch (IOException e) {
			System.err.println(e.getMessage());
		}
	}

	private boolean isPublic(MethodDeclaration md) {
		for(Modifier m: md.getModifiers()) {
			if(m.getKeyword().asString().equals(Constants.PUBLIC)) {
				return true;
			}
		}
		return false;
	}
	
	public void writeSha256ToSootMethodNameFile() {
		File f = new File(String.format("%s/%s.txt", this.destination, "sha256ToSootMethodName"));
		try {
			if (f.createNewFile()) {
				FileWriter fw = new FileWriter(f);
				for (Map.Entry<String, String> entry : this.sha256ToSootMethodName.entrySet()) {
				    fw.write(String.format("%s|||%s\n", entry.getKey(), entry.getValue()));
				}
				fw.close();
			} else {
				System.err.println(String.format("File %s already exists.", f.getName()));
			}
		} catch (IOException e) {
			System.err.println(e.getMessage());
		}
	}
}