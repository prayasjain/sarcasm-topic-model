package experiments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Scanner;
import java.util.StringTokenizer;

import data.LabeledReviews;
import data.QueryLog;
import model.*;
import utils.TopicModelUtils;

public class Whatever {

	// parameters
	static String train;
	static String test;
	static String model;
	static String goal;
	static String testlist;
	static String senti_list = "";
	static double gamma;
	static int samples, burn, step;
	static int Z, Y, X,S;  // latent variable dimensions
	static int[] arr_Z;
	static double au, ad, ac; // alpha hyperparameters
	static double bw, bd, bu, bx; // beta hyperparameters
	static double bernp; //bernoulli hyperparameter
	static int R;
	static double[] arr_bw, arr_ad;
	static double[][] arr_bwz;
	static double[][][] arr_bswz;
	static int num_labels;
	static int words_thresh;
	static int train_on_small;
	static boolean hest;
	public static double ll;

	static QueryLog l;
	static LabeledReviews lr;

	public static void main (String[] args) throws FileNotFoundException {

		System.out.println("Welcome to Aditya's TCS Lab LDA.");

		loadParametersAndData(args);		
		
		long start = System.currentTimeMillis();
		System.out.println("Experiment: estimating  "+model+" model");
		System.out.println("Started at: "+utils.Time.current());

		
		double[][] P_w_z = null;
		double[][][] P_s_w_z = null;

		PrintStream original = System.out;
		
		String myfolder = "Output_"+utils.Time.current().replaceAll(":","").replaceAll(" ","_");
		String dir = "/data/aa1/PhD_Sem3/EclipseOutput/" + myfolder;
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
            dir = dir+"/";
            
		/* Aditya */
            
           
		for(int i=0; i< arr_Z.length; i++)
		{
			Z = arr_Z[i];

			
			String filename = dir+model+"_"+Z+"topics_"+samples+"samples_"+words_thresh+"wordsthresh";
			System.out.println("Starting Estimation. Output will be sent to"+filename+" .");
			System.setOut(new PrintStream(new File(filename)));
			System.out.println("For training corpus = "+train+" and test="+test+" and hest="+hest);
			if (model.equals("LDA2P"))
				P_w_z = (new LDA2P()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z,ad, bw, burn, samples, step, hest, lr);
			if (model.equals("LDA2"))
				P_w_z = (new LDA2()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z, ad, bw, burn, samples, step);
			if (model.equals("LDA2P"))
				P_w_z = (new LDA2P()).estimate(lr.w_ij, lr.W, lr.D, lr.N, Z,ad, bw, burn, samples, step, hest, lr);
			
			if (model.equals("SLDA2")) 
				P_s_w_z = (new SLDA2()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step);
			if (model.equals("LDA2PI"))
				P_w_z = (new LDA2PI()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,ad, bw, burn, samples, step, hest, lr);
			
			if (model.equals("SLDA2PI")) 
			{				
				P_s_w_z = (new SLDA2PI()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step,hest, lr);
				ll = SLDA2PI.leftToRightLikelihood(lr.t_w_ij,P_s_w_z,SLDA2PI.g_s, SLDA2PI.a_sz,R);
				SLDA2PI.getTopicCorrelation();
			}
			if (model.equals("SLDA2PISplit")) 
			{
				P_s_w_z = (new SLDA2PISplit()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step,hest, lr);
				ll = SLDA2PISplit.leftToRightLikelihood(lr.t_w_ij,P_s_w_z,SLDA2PISplit.g_s, SLDA2PISplit.a_z,R);
				SLDA2PISplit.getTopicCorrelation();
			}
			
			if (model.equals("LDACTM1")) 
			{
				P_w_z = (new CTM1()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,ad, bw, burn, samples, step, hest, lr);
				
			}
			
			if (model.equals("SLDA3Split")) 
			{
				P_s_w_z = (new SLDA3Split()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step,hest, lr);
				ll = SLDA3Split.leftToRightLikelihood(lr.t_w_ij,P_s_w_z,SLDA3Split.g_s, SLDA3Split.a_z,R);
			}
			if (model.equals("SLDA2PISplitZFirst")) 
			{
				P_s_w_z = (new SLDA2PISplitZFirst()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step,hest, lr);
				ll = SLDA2PISplitZFirst.leftToRightLikelihood(lr.t_w_ij,P_s_w_z,SLDA2PISplitZFirst.g_s, SLDA2PISplitZFirst.a_z,R);
			}
			
			if (model.equals("SLDA2PISplitSFirst")) 
			{
				P_s_w_z = (new SLDA2PISplitSFirst()).estimate(lr.w_ij, lr.l_w,lr.W, lr.D, lr.N, Z,  lr.S, ad, bw,  gamma,  burn, samples,step,hest, lr);
				ll = SLDA2PISplitSFirst.leftToRightLikelihood(lr.t_w_ij,P_s_w_z,SLDA2PISplitSFirst.g_s, SLDA2PISplitSFirst.a_z,R);
			}
			
			if (P_w_z == null && P_s_w_z == null)
			{
				System.out.println("Check model name. No such model found or no estimate returned!");
				return;
			}

			System.out.println("Experiment: finished estimating "+model+" model");
			System.out.println("Elapsed time: "+(System.currentTimeMillis()-start)/1000 + " seconds");
			System.out.println("Finished at: "+utils.Time.current());
			HashMap<String, Integer> fullLexicon = loadLocalSentiLexicon(testlist);
			
			if(model.startsWith("SLDA"))
			{
				System.out.println("Printing top terms for "+model+" model:");   //Aadi: Print top terms in each topic
				/*
				 * for (int s=0; s<lr.S; s++)
				 
				{
					System.out.println("S="+s);
					if (P_s_w_z[s][0] == null)
						System.out.println("WHHHATTT!");
					TopicModelUtils.printTopTerms(P_s_w_z[s], lr.l_w);
				}
				*/

				TopicModelUtils.printTopTerms(P_s_w_z, lr.l_w);
				for (int s=0; s<lr.S; s++)
				 
				{
					
					System.out.println("S="+s);
					TopicModelUtils.printCountSentiWords(P_s_w_z[s], lr.l_w,fullLexicon);
				}
				System.out.println("Test log-likelihood is: "+ll);
			}
			else if (model.startsWith("LDACTM"))
			{
				
				System.out.println("Printing top terms for "+model+" model:");   //Aadi: Print top terms in each topic

				TopicModelUtils.printTopTerms(P_w_z, lr.l_w);
				TopicModelUtils.printCountSentiWords(P_w_z, lr.l_w,fullLexicon);
				if (R==0)
					System.out.println("R not set");
				
				if (lr.t_w_ij == null)
				{
					System.out.println("Test corpus not found!");
				}
				System.out.println("!@!@!");
			//	ll = CTM1.leftToRightLikelihood(lr.t_w_ij,P_w_z,LDA2PI.a_z,R);
				ll = 0.0d;
				System.out.println("Test log-likelihood is not currently implemented: "+ll);
			}
			else{
				
				System.out.println("Printing top terms for "+model+" model:");   //Aadi: Print top terms in each topic

				TopicModelUtils.printTopTerms(P_w_z, lr.l_w);
				TopicModelUtils.printCountSentiWords(P_w_z, lr.l_w,fullLexicon);
				if (R==0)
					System.out.println("R not set");
				
				if (lr.t_w_ij == null)
				{
					System.out.println("Test corpus not found!");
				}
				System.out.println("!@!@!");
				ll = LDA2PI.leftToRightLikelihood(lr.t_w_ij,P_w_z,LDA2PI.a_z,R);
				System.out.println("Test log-likelihood is: "+ll);
				LDA2PI.getTopicCorrelation();
			}
			
			System.setOut(original);
			LDAIterator li = new LDAIterator();
			System.out.println(Z+"\t"+ll);
		}
		
	}


	


	public static void loadParametersAndData(String[] args) throws FileNotFoundException {

		int num_docs = 0; // Aditya
		int num_test_docs = 0;
		
		num_labels = -1;

		for (String arg: args) {
			String[] s = arg.split("=");
			String param = s[0];
			String val = s[1];

			if (param.equals("train")) train = val;
			if (param.equals("test")) test = val;
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
			if (param.equals("R")) R = Integer.parseInt(val);
			if(param.equals("hest")) hest = Boolean.parseBoolean(val);
			if (param.equals("ac")) ac = Double.parseDouble(val);
			if (param.equals("testlist")) testlist = val;    // Train only on small subset of priors

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
			if (param.equals("num_test_docs")) num_test_docs = Integer.parseInt(val);
			if (param.equals("senti_list")) senti_list = val;
			
			if (param.equals("num_labels")) num_labels = Integer.parseInt(val);
			if (param.equals("words_thresh")) words_thresh = Integer.parseInt(val);
			if (param.equals("bernp")) bernp = Integer.parseInt(val);
			if (param.equals("gamma")) gamma = Double.parseDouble(val);

			if (param.equals("S")) S = Integer.parseInt(val);
		}


		System.out.println("Corpus: "+train);
		if (train == null || model == null) {
			System.err.println("ERROR: missing 'model' or 'train' command line parameter to Training.java ");
			System.err.println("\t example of use:");
			System.err.println("\t\t java experiments.Training model=PTM1 train=/workspace/carman/train/training_data");
			System.err.println("ERROR: Or Is this for Sentiment analysis?");
			System.err.println("\t example of use:");
			System.err.println("\t\t java experiments.goal=senti Training model=PTM1 train=/workspace/carman/train/training_data");
		}


		System.out.println("Experiment: loading training train from query train file: "+train);
		if (!train.startsWith("/") && !goal.equals("senti")) train = System.getProperty("user.dir")+"/"+train;

			lr = new LabeledReviews();

		
		
			lr.load(train, num_docs,num_labels,words_thresh, false, true, false);

			lr.loadTestCorpus(test, false, num_test_docs, num_labels,  words_thresh, false,  true,  false);
		
		
		lr.printStats();
		
		
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

		System.out.println("Experiment: finished loading train.");

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

		if (model.startsWith("senti")){
			System.out.println("\t Number of labels :   "+num_labels);
			System.out.println("\t ac: "+ac);
			System.out.println("\t Remove words lower than : " + words_thresh);
		}
		System.out.println("\t bw: "+bw);
		System.out.println("Test Sentilist at:"+ testlist);
		System.out.println("Sentilist for priors at: "+senti_list);
		System.out.println("Corpus: "+train);
	}

	public static HashMap<String, Integer> loadLocalSentiLexicon(String filename)
	{
		HashMap<String, Integer> hm_senti = new HashMap<String, Integer>();
		
		try {
		
			Scanner s = new Scanner(new File(filename), "UTF-8");
		
			while (s.hasNext())
			{
				String str = s.nextLine();
				StringTokenizer st = new StringTokenizer(str," ");
				int num_of_tokens = st.countTokens();
				
					String word = st.nextToken();
			
					
					int polarity = (int)Float.parseFloat(st.nextToken());
					
					hm_senti.put(word,polarity);
			
			}
			
			
		}
		catch (Exception e) {
			e.printStackTrace();
		}	
		
		return hm_senti;

	
	}


}
