package lu.uni.codoclass.utils;

/*-
 * #%L
 * Difuzer
 * 
 * %%
 * Copyright (C) 2020 Jordan Samhi
 * University of Luxembourg - Interdisciplinary Centre for
 * Security Reliability and Trust (SnT) - TruX - All rights reserved
 *
 * %%
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 2.1 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Lesser Public License for more details.
 * 
 * You should have received a copy of the GNU General Lesser Public
 * License along with this program.  If not, see
 * <http://www.gnu.org/licenses/lgpl-2.1.html>.
 * #L%
 */

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.javatuples.Triplet;

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

/**
 * This class sets the different option for the application
 * @author Jordan Samhi
 *
 */

public class CommandLineOptions {

	private static final Triplet<String, String, String> SOURCE = new Triplet<String, String, String>("source", "s", "Source folder");
	private static final Triplet<String, String, String> ANDROID_JAR = new Triplet<String, String, String>("android_jar", "a", "Android jar path");
	private static final Triplet<String, String, String> OUTPUT = new Triplet<String, String, String>("output", "o", "Destination folder for files");
	private static final Triplet<String, String, String> SOURCE_CODE = new Triplet<String, String, String>("code", "c", "Extract source code");
	private static final Triplet<String, String, String> DOCUMENTATION = new Triplet<String, String, String>("documentation", "d", "Extract documentation");
	private static final Triplet<String, String, String> HELP = new Triplet<String, String, String>("help", "h", "Print this message");

	private Options options, firstOptions;
	private CommandLineParser parser;
	private CommandLine cmdLine, cmdFirstLine;
	
	private static CommandLineOptions instance;
	
	public CommandLineOptions() {
		this.options = new Options();
		this.firstOptions = new Options();
		this.initOptions();
		this.parser = new DefaultParser();
	}
	
	public static CommandLineOptions v() {
		if(instance == null) {
			instance = new CommandLineOptions();
		}
		return instance;
	}
	
	public void parseArgs(String[] args) {
		this.parse(args);
	}

	/**
	 * This method does the parsing of the arguments.
	 * It distinguished, real options and help option.
	 * @param args the arguments of the application
	 */
	private void parse(String[] args) {
		HelpFormatter formatter = null;
		try {
			this.cmdFirstLine = this.parser.parse(this.firstOptions, args, true);
			if (this.cmdFirstLine.hasOption(HELP.getValue0())) {
				formatter = new HelpFormatter();
				formatter.printHelp(Constants.TOOL_NAME, this.options, true);
				System.exit(0);
			}
			this.cmdLine = this.parser.parse(this.options, args);
		} catch (ParseException e) {
			System.err.println(e.getMessage());
			System.exit(1);
		}
	}

	/**
	 * Initialization of all recognized options
	 */
	private void initOptions() {
		final Option source = Option.builder(SOURCE.getValue1())
				.longOpt(SOURCE.getValue0())
				.desc(SOURCE.getValue2())
				.hasArg(true)
				.argName(SOURCE.getValue0())
				.required(true)
				.build();
		
		final Option android_jar = Option.builder(ANDROID_JAR.getValue1())
				.longOpt(ANDROID_JAR.getValue0())
				.desc(ANDROID_JAR.getValue2())
				.hasArg(true)
				.argName(ANDROID_JAR.getValue0())
				.required(true)
				.build();

		final Option output = Option.builder(OUTPUT.getValue1())
				.longOpt(OUTPUT.getValue0())
				.desc(OUTPUT.getValue2())
				.hasArg(true)
				.argName(OUTPUT.getValue0())
				.required(false)
				.build();
		
		final Option source_code = Option.builder(SOURCE_CODE.getValue1())
				.longOpt(SOURCE_CODE.getValue0())
				.desc(SOURCE_CODE.getValue2())
				.hasArg(false)
				.argName(SOURCE_CODE.getValue0())
				.required(false)
				.build();
		
		final Option documentation = Option.builder(DOCUMENTATION.getValue1())
				.longOpt(DOCUMENTATION.getValue0())
				.desc(DOCUMENTATION.getValue2())
				.hasArg(false)
				.argName(DOCUMENTATION.getValue0())
				.required(false)
				.build();

		final Option help = Option.builder(HELP.getValue1())
				.longOpt(HELP.getValue0())
				.desc(HELP.getValue2())
				.argName(HELP.getValue0())
				.build();
		

		this.firstOptions.addOption(help);

		this.options.addOption(source);
		this.options.addOption(output);
		this.options.addOption(documentation);
		this.options.addOption(source_code);
		this.options.addOption(android_jar);

		for(Option o : this.firstOptions.getOptions()) {
			this.options.addOption(o);
		}
	}
	
	public String getSource() {
		return cmdLine.getOptionValue(SOURCE.getValue0());
	}
	
	public String getOutput() {
		return cmdLine.getOptionValue(OUTPUT.getValue0());
	}
	
	public String getAndroidJar() {
		return cmdLine.getOptionValue(ANDROID_JAR.getValue0());
	}

	public boolean hasOutput() {
		return this.cmdLine.hasOption(OUTPUT.getValue1());
	}
	
	public boolean hasDocumentation() {
		return this.cmdLine.hasOption(DOCUMENTATION.getValue1());
	}
	
	public boolean hasSourceCode() {
		return this.cmdLine.hasOption(SOURCE_CODE.getValue1());
	}
}
