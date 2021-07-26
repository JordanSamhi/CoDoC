package lu.uni.cobaclass;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.codec.digest.DigestUtils;

import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.expr.AnnotationExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.JavaParserTypeSolver;
import com.github.javaparser.symbolsolver.resolution.typesolvers.ReflectionTypeSolver;
import com.github.javaparser.utils.CodeGenerationUtils;
import com.github.javaparser.utils.SourceRoot;

import lu.uni.cobaclass.utils.CommandLineOptions;
import lu.uni.cobaclass.utils.Constants;
import lu.uni.cobaclass.utils.Utils;
import soot.SootClass;
import soot.SootMethod;

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
	private Map<SootMethod, String> methodToSourceCode;
	private Map<SootMethod, String> methodToDocumentation;
	private Map<SootMethod, String> abstractMethodToDocumentation;
	private CombinedTypeSolver combinedTypeSolver;
	private Map<SootMethod, String> sha256ToSootMethodName;
	private List<SootMethod> methodsWithNoDocumentation;

	public SourceCodeExtractor(String pathToSourceRootFolder) {
		this.pathToSourceRootFolder = pathToSourceRootFolder;
		this.sourceRoot = new SourceRoot(CodeGenerationUtils.mavenModuleRoot(SourceCodeExtractor.class).resolve(this.pathToSourceRootFolder));
		this.methodToSourceCode = new HashMap<SootMethod, String>();
		this.methodToDocumentation = new HashMap<SootMethod, String>();
		this.abstractMethodToDocumentation = new HashMap<SootMethod, String>();
		this.combinedTypeSolver = new CombinedTypeSolver();
		this.combinedTypeSolver.add(new ReflectionTypeSolver());
		this.methodsWithNoDocumentation = new ArrayList<SootMethod>();
		combinedTypeSolver.add(new JavaParserTypeSolver(new File(CommandLineOptions.v().getSource())));
		this.sha256ToSootMethodName = new HashMap<SootMethod, String>();
	}

	public void parseClass(String className) {
		this.cu = this.sourceRoot.parse("", className);
	}

	public void extractMethodArtefacts(SootMethod sm) {
		if(this.cu == null) {
			throw new ExceptionInInitializerError("Compilation Unit not initialized, please run parseClass before.");
		}
		cu.accept(new VoidVisitorAdapter<Void>() {
			public void visit(MethodDeclaration md, Void arg) {
				if(Utils.sootMethodEqualsNormalMethod(sm, md)) {
					if(md.getBody().isPresent()) {
						StringBuilder source_code = new StringBuilder();
						source_code.append(md.getDeclarationAsString(true, true, true));
						source_code.append(md.getBody().get());
						methodToSourceCode.put(sm, source_code.toString());
						String sha256SootSignature = DigestUtils.sha256Hex(sm.getSignature());
						sha256ToSootMethodName.put(sm, sha256SootSignature);
						if(md.getComment().isPresent()) {
							methodToDocumentation.put(sm, md.getComment().get().getContent());
						}else {
							if(hasOverride(md)) {
								methodsWithNoDocumentation.add(sm);
							}
						}
					}else {
						if(md.getComment().isPresent()) {
							abstractMethodToDocumentation.put(sm, md.getComment().get().getContent());
						}
					}
				}
			}
		}, null);
	}

	public void dump() {
		String parentDocumentation = null;
		for(SootMethod sm: this.methodsWithNoDocumentation) {
			if(!this.methodToDocumentation.containsKey(sm)) {
				parentDocumentation = getParentDocumentation(sm);
				if(parentDocumentation != null) {
					this.methodToDocumentation.put(sm, parentDocumentation);
				}
			}
		}
		String destination = null;
		if(CommandLineOptions.v().hasOutput()) {
			destination = CommandLineOptions.v().getOutput();
		}else {
			destination = Constants.DEFAULT_OUTPUT;
		}
		if(CommandLineOptions.v().hasDocumentation()) {
			for (Map.Entry<SootMethod, String> entry : this.methodToDocumentation.entrySet()) {
				writeInFile(destination, "documentation", this.sha256ToSootMethodName.get(entry.getKey()), entry.getValue());
			}
		}
		if(CommandLineOptions.v().hasSourceCode()) {
			for (Map.Entry<SootMethod, String> entry : this.methodToSourceCode.entrySet()) {
				if(this.methodToDocumentation.containsKey(entry.getKey())){
					writeInFile(destination, "source_code", this.sha256ToSootMethodName.get(entry.getKey()), entry.getValue());
				}
			}
		}
		this.writeSha256ToSootMethodNameFile(destination);
	}

	private String getParentDocumentation(SootMethod sm) {
		String subsig = sm.getSubSignature();
		SootClass parentClass = sm.getDeclaringClass().getSuperclass();
		SootMethod parentMethod = parentClass.getMethodUnsafe(subsig);
		while(parentMethod == null) {
			if(!parentClass.hasSuperclass()) {
				break;
			}
			parentClass = parentClass.getSuperclass();
			parentMethod = parentClass.getMethodUnsafe(subsig);
		}
		if(parentMethod != null) {
			if(this.methodToDocumentation.containsKey(parentMethod)){
				return methodToDocumentation.get(parentMethod);
			}else if(this.abstractMethodToDocumentation.containsKey(parentMethod)){
				return abstractMethodToDocumentation.get(parentMethod);
			}
		}
		return null;
	}

	private boolean hasOverride(MethodDeclaration md) {
		for(AnnotationExpr ae: md.getAnnotations()) {
			if(ae.getNameAsString().equals("Override")) {
				return true;
			}
		}
		return false;
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

	public void writeSha256ToSootMethodNameFile(String destination) {
		File f = new File(String.format("%s/%s.txt", destination, "sha256ToSootMethodName"));
		try {
			if (f.createNewFile()) {
				FileWriter fw = new FileWriter(f);
				for (Map.Entry<SootMethod, String> entry : this.sha256ToSootMethodName.entrySet()) {
					if(this.methodToDocumentation.containsKey(entry.getKey()) && this.methodToSourceCode.containsKey(entry.getKey())) {
						fw.write(String.format("%s|||%s\n", entry.getKey().getSignature(), entry.getValue()));
					}
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