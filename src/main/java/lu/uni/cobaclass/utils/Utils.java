package lu.uni.cobaclass.utils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.type.Type;
import com.github.javaparser.resolution.types.ResolvedType;
import com.github.javaparser.symbolsolver.javaparsermodel.JavaParserFacade;
import com.github.javaparser.symbolsolver.resolution.typesolvers.CombinedTypeSolver;
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

public class Utils {

	public static String dotClassNametoSlashClassName(String className) {
		String newClassName = className.replace(".", "/");
		if(newClassName.contains("$")) {
			return newClassName.split("\\$")[0];
		}
		return newClassName;
	}

	public static String slashClassNametoDotClassName(String className) {
		return className.replace("/", ".");
	}

	public static String removeLessAndGreaterSigns(String s) {
		if(!s.contains("<")) {
			return s;
		}
		String stack = "";
		char c;
		int idxStart = 0;
		int idxEnd = 0;
		for(int i = 0 ; i < s.length() ; i++) {
			c = s.charAt(i);
			if(c == '<') {
				if(stack.isEmpty()) {
					idxStart = i;
				}
				stack += "x";
			}else if(c == '>') {
				stack = stack.substring(0, stack.length() - 1);
				if(stack.isEmpty()) {
					idxEnd = i;
					break;
				}
			}
		}
		String newS = String.format("%s%s", s.subSequence(0, idxStart), s.subSequence(idxEnd + 1, s.length()));
		return Utils.removeLessAndGreaterSigns(newS);
	}

	public static void getJavaFilesPaths(File file, List<File> javaFiles) {
		File[] files = file.listFiles();
		for(File f: files) {
			if(f.isDirectory()) {
				Utils.getJavaFilesPaths(f, javaFiles);
			}else if(f.isFile()) {
				if(f.getName().endsWith(".java")) {
					if(!javaFiles.contains(f)) {
						javaFiles.add(f);
					}
				}
			}
		}
	}

	public static String fullPathToClassName(String fullPath) {
		String path = Utils.slashClassNametoDotClassName(fullPath);
		String[] split = path.split("\\.");
		int idx = 0;
		StringBuilder sb = new StringBuilder();
		for(int i = 0 ; i < split.length ; i++) {
			if(split[i].equals(Constants.CURRENT_SOURCE_CODE_FOLDER)) {
				idx = i;
			}
			if(idx != 0 && i > idx && i != split.length - 1) {
				if(sb.length() != 0) {
					sb.append(".");
				}
				sb.append(split[i]);
			}
		}
		return sb.toString();
	}
	
	public static String resolveParam(Type type, CombinedTypeSolver combinedTypeSolver) {
		ResolvedType rt = null,
				rt1 = null;
		String param = null;
		if(type.isReferenceType()) {
			try {
				rt = JavaParserFacade.get(combinedTypeSolver).convertToUsage(type);
			}catch(Exception e) {
				System.err.println(e.getMessage());
			}
			if(rt != null) {
				if(rt.isReferenceType()) {
					param = rt.asReferenceType().getQualifiedName();
				}else if(rt.isArray()) {
					rt1 = rt.asArrayType().getComponentType();
					if(rt1.isReferenceType()) {
						param = rt1.asReferenceType().getQualifiedName();							
					}else {
						param = type.toString();
					}
				}else if(rt.isTypeVariable()) {
					param = rt.asTypeVariable().qualifiedName();
				}
			}else {
				param = type.toString();
			}
		}else {
			param = type.toString();
		}
		return param;
	}

	public static String toSootClassName(String className, MethodDeclaration methodDeclaration, CombinedTypeSolver combinedTypeSolver) {
		List<String> params = new ArrayList<String>();
		String param = null;
		Type type = null;
		NodeList<Parameter> parameters = methodDeclaration.getParameters();
		for(Parameter p: parameters) {
			type = p.getType();
			param = Utils.resolveParam(type, combinedTypeSolver);
			if((p.toString().contains("...") || p.toString().contains("[]")) && !param.contains("[]")) {
				param = String.format("%s[]", param);
			}
			params.add(param);
		}
		String ret = Utils.resolveParam(methodDeclaration.getType(), combinedTypeSolver);
		String methodName = methodDeclaration.getName().asString();
		return String.format("<%s: %s %s(%s)>", className, ret, methodName, String.join(",", params));
	}
}
