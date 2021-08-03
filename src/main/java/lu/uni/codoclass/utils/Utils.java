package lu.uni.codoclass.utils;

import java.util.ArrayList;
import java.util.List;

import com.github.javaparser.ast.NodeList;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;

/*-
 * #%L
 * CoDoClaSS
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
import soot.SootMethod;
import soot.Type;

public class Utils {

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

	public static String sootClassNameToJavaPath(String sootClassName) {
		StringBuilder sb = new StringBuilder(sootClassName.split("\\$")[0].replace(".", "/"));
		sb.append(".java");
		return sb.toString();
	}

	public static String fqnToNormal(String s) {
		String separator = null;
		if(s.contains("$")) {
			separator = "\\$";
		}else {
			separator = "\\.";
		}
		String[] split = s.split(separator);
		return split[split.length - 1];
	}

	public static List<String> paramListToString(NodeList<Parameter> parameters) {
		List<String> params = new ArrayList<String>();
		String param = null;
		for(Parameter p: parameters) {
			param = Utils.removeLessAndGreaterSigns(p.getTypeAsString());
			if(p.toString().contains("...")) {
				param = String.format("%s[]", param);
			}
			params.add(param);
		}
		return params;
	}

	public static List<String> sootParamListToString(List<Type> types) {
		List<String> params = new ArrayList<String>();
		for(Type t: types) {
			params.add(Utils.fqnToNormal(t.toString()));
		}
		return params;
	}

	public static boolean sootMethodEqualsNormalMethod(SootMethod sm, MethodDeclaration md) {
		String currentMethodRet = Utils.removeLessAndGreaterSigns(md.getTypeAsString());
		String currentMethodName = md.getNameAsString();
		String sootMethodRet = Utils.fqnToNormal(sm.getReturnType().toString());
		String sootMethodName = sm.getName();
		List<String> currentMethodParams = Utils.paramListToString(md.getParameters());
		List<String> sootMethodParams = Utils.sootParamListToString(sm.getParameterTypes());
		if(currentMethodRet.equals(sootMethodRet) 
				&& currentMethodName.equals(sootMethodName)
				&& Utils.areListsEquals(currentMethodParams, sootMethodParams)) {
			return true;
		}
		return false;
	}

	public static boolean areListsEquals(List<String> l1, List<String> l2) {
		int l1Size = l1.size();
		int l2Size = l2.size();
		boolean equals = false;
		if(l1Size == l2Size) {
			equals = true;
			for(int i = 0 ; i < l1Size ; i++) {
				if(!l1.get(i).equals(l2.get(i))) {
					equals = false;
					break;
				}
			}
		}
		return equals;
	}
}