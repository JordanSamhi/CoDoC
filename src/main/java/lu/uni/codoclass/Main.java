package lu.uni.codoclass;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Map;

import lu.uni.codoclass.utils.CommandLineOptions;
import lu.uni.codoclass.utils.Constants;
import lu.uni.codoclass.utils.Utils;
import soot.G;
import soot.PackManager;
import soot.Scene;
import soot.SceneTransformer;
import soot.SootClass;
import soot.SootMethod;
import soot.Transform;
import soot.options.Options;
import soot.util.Chain;

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
		String android_jar = CommandLineOptions.v().getAndroidJar();

		initializeSoot(android_jar);
		PackManager.v().getPack("wjtp").add(
				new Transform("wjtp.myTransform", new SceneTransformer() {
					protected void internalTransform(String phaseName, @SuppressWarnings("rawtypes") Map options) {
						Chain<SootClass> classes = Scene.v().getClasses();
						String javaFilePath = null;
						SourceCodeExtractor sce = new SourceCodeExtractor(source_folder);
						for(SootClass sc: classes) {
							if(sc.isPublic()) {
								javaFilePath = String.format("%s/%s", source_folder, Utils.sootClassNameToJavaPath(sc.getName()));
								Path path = Paths.get(javaFilePath);
								if(Files.exists(path)) {
									sce.parseClass(javaFilePath);
									for(SootMethod sm: sc.getMethods()) {
										if(sm.isPublic() && !sm.isConstructor()) {
											sce.extractMethodArtefacts(sm);
										}
									}
								}
							}
						}
						sce.dump();
					}
				}));

		PackManager.v().runPacks();
	}

	private static void initializeSoot(String jar) {
		G.reset();
		Options.v().setPhaseOption("cg", "enabled:false");
		Options.v().set_allow_phantom_refs(true);
		Options.v().set_output_format(Options.output_format_none);
		Options.v().set_whole_program(true);
		List<String> dirs = new ArrayList<String>();
		dirs.add(jar);
		Options.v().set_process_dir(dirs);
		Scene.v().loadNecessaryClasses();
	}
}
