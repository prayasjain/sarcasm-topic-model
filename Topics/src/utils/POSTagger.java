package utils;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.List;

import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.process.CoreLabelTokenFactory;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.process.PTBTokenizer;
import edu.stanford.nlp.process.TokenizerFactory;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

/** This demo shows user-provided sentences (i.e., {@code List<HasWord>})
 *  being tagged by the tagger. The sentences are generated by direct use
 *  of the DocumentPreprocessor class.
 *
 *  @author Christopher Manning
 */
/*
 * Modified by Aditya J.
 * 
 * Input: 1) model file, 2) Directory where data files are present
 * 
 * 
 * 1) model file = C:\Aditya\POSTaggerModels\english-left3words-distsim.tagger
 * Output goes to directory same as (2) with "input" replaced with "output". This directory must be created
 * beforehand. All files are copied with a ".wordpos" extension.
 * Output format is word/POS.
 */
class POSTagger {

	

	
	public static void main(String[] args) throws Exception 
	{
		
		File[] faFiles = new File(args[1]).listFiles();
		String output = "";

		if (faFiles == null || faFiles.length <= 0){
			System.err.println("Empty or not a directory");
			return;
		}
		/*
		 * Create output directory if it does not exist
		 */
		File outputDir = new File(args[1].replace("input", "output"));
		if (!outputDir.exists()) {
			System.out.println("Creating directory: " + outputDir.getAbsolutePath());
			boolean result = outputDir.mkdir();  

			if(result) {    
				System.out.println("DIR created");  
			}
		}
		int count = 0;
		POSTagger pt;
		pt = new POSTagger();
		String modelFile = args[0];
		for(File file: faFiles){

			output = "";
			String outputPath = file.getAbsolutePath().replace("input","output");
			FileWriter p = new FileWriter(outputPath+".wordpos");

			BufferedWriter bw = new BufferedWriter(p);
			if(file.isDirectory()){
				;
			}

			if(file.getName().matches("^(.*?)")){
				System.out.println("Processing "+ file.getAbsolutePath());
			}

			MaxentTagger tagger = new MaxentTagger(modelFile);
			TokenizerFactory<CoreLabel> ptbTokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(),
					"untokenizable=noneKeep");
			BufferedReader r = new BufferedReader(new InputStreamReader(new FileInputStream(file.getAbsolutePath()), "utf-8"));

			//PrintWriter pw = new PrintWriter(new OutputStreamWriter(System.out, "utf-8"));
			DocumentPreprocessor documentPreprocessor = new DocumentPreprocessor(r);
			documentPreprocessor.setTokenizerFactory(ptbTokenizerFactory);
			for (List<HasWord> sentence : documentPreprocessor) {
				List<TaggedWord> tSentence = tagger.tagSentence(sentence);
				bw.write(Sentence.listToString(tSentence, false)+"\n");
				bw.newLine();
				output += Sentence.listToString(tSentence, false)+"\n";
			}

			count++;

			// bw.write(output);
			bw.close();
			p.close();
		}

		System.out.println("Total : " +count);
	}
}
