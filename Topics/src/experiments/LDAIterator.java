package experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.StringTokenizer;

import model.*;
import utils.TopicModelUtils;
import data.LabeledReviews;
import data.QueryLog;

public class LDAIterator {

	// parameters
	static String log;
	static String model;
	static String goal;

	static double gamma;
	static int samples, burn, step;
	static int Z, Y, X,S;  // latent variable dimensions
	static int[] arr_Z;
	static double au, ad, ac; // alpha hyperparameters
	static double bw, bd, bu, bx; // beta hyperparameters
	static double bernp; //bernoulli hyperparameter

	static double[] arr_bw, arr_ad;
	static int num_labels;
	static int words_thresh;
	public static double ll;
	
	static QueryLog l;
	static LabeledReviews lr;

	public static void main (String[] args) throws FileNotFoundException {


		loadParametersAndData(args);		

		long start = System.currentTimeMillis();
		System.out.println("Experiment: estimating  "+model+" model");
		System.out.println("Started at: "+utils.Time.current());

		double[][] P_w_z = null;
		double[][][] P_s_w_z = null;

		PrintStream original = System.out;
		
		String myfolder = "Output_"+utils.Time.current().replaceAll(".","").replaceAll(":","").replaceAll(" ","_");
		String dir = "C:\\Aditya\\Output\\" + myfolder;
        boolean result = false;
        File directory = new File(dir);

        if (!directory.exists()) {
            result = directory.mkdir();

            if (result) {
                System.out.println("Output Folder created as: "+dir);
                
            } else {
                System.out.println("Failed creating output folder");
            }
        }
            dir = dir+"\\";
            
		/* Aditya */
		for(int i=0; i< arr_Z.length; i++)
		{
			Z = arr_Z[i];
			String filename = dir+model+"_"+Z+"topics_"+samples+"samples_"+words_thresh+"wordsthresh";
			System.out.println(filename+" Done.");
			System.setOut(new PrintStream(new File(filename)));
			if (model.equals("LDA") && goal.equals("senti"))
				P_w_z = (new LDA()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, ad, bw, burn, samples, step);
			else if (model.equals("LDA"))
				P_w_z = (new LDA()).estimate(l.projectOverUsers(), l.W, l.D, l.N, Z, ad, bw, burn, samples, step);

			if (model.equals("LDA2") && goal.equals("senti"))
				P_w_z = (new LDA2()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, ad, bw, burn, samples, step);
			else if (model.equals("LDA2"))
				P_w_z = (new LDA2()).estimate(l.projectOverUsers(), l.W, l.D, l.N, Z, ad, bw, burn, samples, step);

			if (model.equals("sentidoc2")) P_w_z = (new LDAWithDocLabels2()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, lr.hm_sentiwordlist, lr.l_w, lr.s_i, num_labels, ad, bw, ac, burn, samples, step);
			//	if (model.equals("sentis1")) P_w_z = (new LDAWithSwitch1()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z,  lr.l_w, ad, bw, ad, bw, ad, burn, samples, step);
			if (model.equals("sentence1")) P_w_z = (new LDASentenceSwitch1()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, lr.l_w, ad, bw, bernp, burn, samples, step);
			if (model.equals("sentidoc1")) P_w_z = (new LDAWithDocLabels1()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, lr.hm_sentiwordlist, lr.l_w, lr.s_i, num_labels, ad, bw, ac, burn, samples, step);
			if (model.equals("sentii")) P_w_z = (new LDAWithSentiInit()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, lr.hm_sentiwordlist, lr.l_w, ad, bw, burn, samples, step);
			if (model.equals("LDAWithSentiPrior1")) P_w_z = (new LDAWithSentiPrior1()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, lr.hm_sentiwordlist, lr.l_w, ad, bw, burn, samples, step);
			if (model.equals("LDAWithSentiPrior2")) P_w_z = (new LDAWithSentiPrior2()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, lr.hm_sentiwordlist, lr.l_w, ad, bw, burn, samples, step);
			if (model.equals("SLDA1")) P_s_w_z = (new SLDA1()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step);

			if (model.equals("SLDA2")) P_s_w_z = (new SLDA2()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step);
			if (model.equals("SLDA3")) P_s_w_z = (new SLDA3()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step);

			if (model.equals("SLDA4")) P_s_w_z = (new SLDA4()).estimate(lr.w_ij, lr.s_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step);

			/* Aditya end */

			if (model.equals("PTM1")) P_w_z = (new PTM1()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, Z, ad, bu, bw, burn, samples, step);
			if (model.equals("PTM2")) P_w_z = (new PTM2()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, Z, au, bd, bw, burn, samples, step);
			if (model.equals("PTM3")) P_w_z = (new PTM3()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, Y, Z, au, ad, bw, burn, samples, step);
			if (model.equals("PTM4")) P_w_z = (new PTM4()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, X, Y, Z, au, ad, bx, bw, burn, samples, step);

			if (model.equals("PTM1u")) P_w_z = (new PTM1_user()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, Z, ad, bu, bw, burn, samples, step);
			if (model.equals("PTM2u")) P_w_z = (new PTM2_user()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, Z, au, bd, bw, burn, samples, step);
			if (model.equals("PTM3s")) P_w_z = (new PTM3_sequential()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, Y, Z, au, ad, bw, burn, samples, step);
			if (model.equals("PTM4s")) P_w_z = (new PTM4_sequential()).estimate(l.w_uij, l.d_ui, l.W, l.D, l.N, X, Y, Z, au, ad, bx, bw, burn, samples, step);

			if (model.equals("LDAWithSwitch2")) P_s_w_z = (new LDAWithSwitch2()).estimate(lr.w_ij,lr.W, lr.D, lr.N, lr.S, arr_Z, arr_ad, arr_bw, gamma, burn, samples, step  );

			if(model.startsWith("SLDA") || model.startsWith("LDAWithSwitch"))
			{
				System.out.println("Experiment: finished estimating "+model+" model");
				System.out.println("Elapsed time: "+(System.currentTimeMillis()-start)/1000 + " seconds");
				System.out.println("Finished at: "+utils.Time.current());

				System.out.println();
				System.out.println();
				System.out.println("Printing top terms for "+model+" model:");   //Aadi: Print top terms in each topic
				for (int s=0; s<lr.S; s++)
				{
					System.out.println("S="+s);
					TopicModelUtils.printTopTerms(P_s_w_z[s], lr.l_w);
				}

				for (int s=0; s<lr.S; s++)
				{
					System.out.println("S="+s);
					TopicModelUtils.printCountSentiWords(P_s_w_z[s], lr.l_w,lr.hm_sentiwordlist);
				}
			}
			else if (P_w_z == null)
			{
				System.out.println("Check model name. No such model found or no estimate returned!");
				return;
			}
			else
			{
				System.out.println("Experiment: finished estimating "+model+" model");
				System.out.println("Elapsed time: "+(System.currentTimeMillis()-start)/1000 + " seconds");
				System.out.println("Finished at: "+utils.Time.current());

				System.out.println();
				System.out.println();
				System.out.println("Printing top terms for "+model+" model:");   //Aadi: Print top terms in each topic
				if (goal.equals("senti"))
					TopicModelUtils.printTopTerms(P_w_z, lr.l_w);
				else
					TopicModelUtils.printTopTerms(P_w_z,l.loadLexicon());

				if (model.contains("senti") || model.contains("Senti")|| goal.equals("senti"))
					TopicModelUtils.printCountSentiWords(P_w_z, lr.l_w,lr.hm_sentiwordlist);
			}
			
			System.setOut(original);
			System.out.println(Z+"\t"+ll);
		}

	}


	public static void loadParametersAndData(String[] args) throws FileNotFoundException {

		int num_docs = 0; // Aditya
		String senti_list = "";
		num_labels = -1;

		for (String arg: args) {
			String[] s = arg.split("=");
			String param = s[0];
			String val = s[1];

			if (param.equals("log")) log = val;
			if (param.equals("goal")) goal = val;
			if (param.equals("model")) model = val;
			if (param.equals("step")) step = Integer.parseInt(val);
			if (param.equals("burn")) burn = Integer.parseInt(val);
			if (param.equals("samples")) samples = Integer.parseInt(val);
			if (param.equals("Z")) Z = Integer.parseInt(val);
			if (param.equals("Y")) Y = Integer.parseInt(val);
			if (param.equals("X")) X = Integer.parseInt(val);
			if (param.equals("ad")) ad = Double.parseDouble(val);
			if (param.equals("au")) au = Double.parseDouble(val);
			if (param.equals("bw")) bw = Double.parseDouble(val);
			if (param.equals("bu")) bu = Double.parseDouble(val);
			if (param.equals("bd")) bd = Double.parseDouble(val);
			if (param.equals("bx")) bx = Double.parseDouble(val);
			if (param.equals("ac")) ac = Double.parseDouble(val);

			if (param.equals("arr_ad"))
			{
				String arr_ad_val = val;

				StringTokenizer st = new StringTokenizer(val,",");
				arr_ad = new double[st.countTokens()];
				int i = 0;
				while (st.hasMoreTokens())
				{
					arr_ad[i] = Double.parseDouble(st.nextToken());
					i++;
				}
			}

			if (param.equals("arr_Z"))
			{
				String arr_ad_val = val;

				StringTokenizer st = new StringTokenizer(val,",");
				arr_Z = new int[st.countTokens()];
				int i = 0;
				while (st.hasMoreTokens())
				{
					arr_Z[i] = Integer.parseInt(st.nextToken());
					i++;
				}
			}
			if (param.equals("arr_bw"))
			{
				String arr_bw_val = val;

				StringTokenizer st = new StringTokenizer(val,",");
				arr_bw = new double[st.countTokens()];
				int i = 0;
				while (st.hasMoreTokens())
				{
					arr_bw[i] = Double.parseDouble(st.nextToken());
					i++;
				}
			}
			if (param.equals("num_docs")) num_docs = Integer.parseInt(val);
			if (param.equals("senti_list")) senti_list = val;
			if (param.equals("num_labels")) num_labels = Integer.parseInt(val);
			if (param.equals("words_thresh")) words_thresh = Integer.parseInt(val);
			if (param.equals("bernp")) bernp = Integer.parseInt(val);
			if (param.equals("gamma")) gamma = Double.parseDouble(val);

			if (param.equals("S")) S = Integer.parseInt(val);
		}


		System.out.println("Corpus: "+log);
		if (log == null || model == null) {
			System.err.println("ERROR: missing 'model' or 'log' command line parameter to Training.java ");
			System.err.println("\t example of use:");
			System.err.println("\t\t java experiments.Training model=PTM1 log=/workspace/carman/data/training_data");
			System.err.println("ERROR: Or Is this for Sentiment analysis?");
			System.err.println("\t example of use:");
			System.err.println("\t\t java experiments.goal=senti Training model=PTM1 log=/workspace/carman/data/training_data");
		}


		System.out.println("Experiment: loading training data from query log file: "+log);
		if (!log.startsWith("/") && !goal.equals("senti")) log = System.getProperty("user.dir")+"/"+log;

		if (goal.equals("senti"))
		{	lr = new LabeledReviews();

		if (model.equals("SLDA1"))
			lr.load(log, num_docs,num_labels,words_thresh, true, true, false);
		else if (model.equals("SLDA2"))
			lr.load(log, num_docs,num_labels,words_thresh, false, true, false);

		if (model.equals("SLDA3"))
			lr.load(log, num_docs,num_labels,words_thresh, true, true, false);

		if (model.equals("SLDA4"))
			lr.load(log, num_docs,num_labels,words_thresh, true, true, true);
		else
			lr.load(log, num_docs, S, words_thresh, false, true, false);

		lr.printStats();
		}
		else
		{
			l = new QueryLog();
			l.load(log);
			l.printStats();
		}

		if (model.equals("sentidoc1") && num_labels == -1)
		{
			System.err.println("Number of labels must be specified in case of document level SA.");
			return;
		}

		if (lr!=null && senti_list!=null && !senti_list.trim().equals(""))
		{
			goal = "senti";
			lr.loadSentiLexicon(senti_list);
		}

		System.out.println("Experiment: finished loading log.");

		// default parameter settings:
		if (step==0) step = 10;
		if (burn==0) burn = 400;
		if (samples==0) samples = 800;
		if (Z==0)    Z = 100;
		if (Y==0)    Y = 10;
		if (X==0)    X = 50;
		if (ad==0.0) ad = 50.0;
		if (au==0.0) au = 50.0;
		if (ac==0.0) ac = 50.0;
		if (bernp==0.0) bernp = 1;

		if (gamma==0.0) gamma = 50.0;
		if (goal.equals("senti"))
		{
			if (bw == 0.0) bw = 0.1 * lr.W;
			if (bd==0.0) bd = 0.1*lr.D;

		}
		else
		{
			if (bw==0.0) bw = 0.1*l.W;
			if (bu==0.0) bu = 0.1*l.U;
			if (bd==0.0) bd = 0.1*l.D;
		}

		if (bx==0.0) bx = 0.1*X;

		printParameterSettings();

	}


	public static void printParameterSettings() {
		System.out.println("Parameter settings:");

		System.out.println("\t model: "+model);
		System.out.println("\t burnIn: "+burn);
		System.out.println("\t samples: "+samples);
		System.out.println("\t step: "+step);
		System.out.println("\t Z:  "+Z);
		System.out.println("\t Number of documents: "+ lr.w_ij.length );
		System.out.println("\t Vocabulary length: "+ lr.l_w.length);
		if (model.startsWith("LDA")) {
			System.out.println("\t ad: "+ad);
		}
		if (model.startsWith("PTM1")) {
			System.out.println("\t ad: "+ad);
			System.out.println("\t bu: "+bu);
		}
		if (model.startsWith("PTM2")) {
			System.out.println("\t au: "+au);
			System.out.println("\t bd: "+bd);
		}
		if (model.startsWith("PTM3")) {
			System.out.println("\t Y:  "+Y);
			System.out.println("\t ad: "+ad);
			System.out.println("\t au: "+au);
		}
		if (model.startsWith("PTM4")) {
			System.out.println("\t Y:  "+Y);
			System.out.println("\t X:  "+X);
			System.out.println("\t ad: "+ad);
			System.out.println("\t au: "+au);
			System.out.println("\t bx: "+bx);
		}
		if (model.startsWith("senti")){
			System.out.println("\t Number of labels :   "+num_labels);
			System.out.println("\t ac: "+ac);
			System.out.println("\t Remove words lower than : " + words_thresh);
		}
		System.out.println("\t bw: "+bw);
	}


}
