package model;

import experiments.Training;
import utils.Gamma;
import utils.TopicModelUtils;

/*
 * Includes sentence labels. Samples a sentence-level sentiment and sentence-level topic first.
 * Then generates the word.
 * 
 */

public class SLDA3 {

	public int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public long N; // total word occurrences

	public double a; // topic smoothing hyperparameter \alpha
	public double b; // term smoothing hyperparameter \beta 

	/* HE */
	//	public double[] a_z;	// a_z[i] = alpha for the i'th topic
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document. Same for alll words in a sentence.

	//	public int[][] N_zd; // N_zd[z][d] = count of z'th topic in d'th document
	//	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	//	public int[] N_z;    // N_z[z] = count of z'th topic
	public int[] N_d;    // N_d[d] = length of document d

	//	public double[][] P_w_z;
	//	public double[][] P_z_d;

	public int num_samples;


	/* New variables for this file */
	public int S;		// No. of sentiment labels

	public int[][] s_di; // s_di[d][i] = sentiment assignment to i'th position in d'th document. Same for all words in a sentence.

	/* For gamma */
	public double g;	// sentiment smoothing hyperparameter \gamma

	/* For pi */
	public int[][] N_sd;	// N_sd[s][d] = count of s'th label in d'th document
	public int[] N_s;		// N_s[s] = count of s'th label

	/* For beta */
	public double[] b_s; // b_s[s] = beta parameter for s'th sentiment label

	/* For new phi */

	public int[][][] N_swz; // N_swz[s][w][z] = count of w'th word for z'th topic with s'th sentiment
	public int[][] N_sw; // N_sw[s][w] = count of w'th word with s'th sentiment


	/* For new theta */
	public int[][][] N_szd; // N_szd[s][z][d] = count of s'th sentiment, z'th topic in d'th document
	public int[][] N_sz; // N_sz[s][z] = count of s'th sentiment in z'th topic

	/* For new alpha */
	public double[][] a_sz; // a_sz[s][z] = alpha for s'th sentiment and z'th topic

	public double[][][] P_s_w_z;
	public double[][][] P_s_z_d;

	//??	public int[][] N_sz; // N_sz[s][z] = count of 


	public double[][][] estimate(int[][] w_di, String[] l_w,int W, int D, long N, int Z, int S, double a, double b, double g, int burnIn, int samples, int step) {

		System.out.println("Estimation begins..");
		this.w_di = w_di;
		this.S = S;
		this.g = g;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		this.a = a;
		this.b = b;
		num_samples = 0;

		//* a_on_Z = a/Z;

		/*
		 * ll_old records ll value in last iteration. Repeat indicates stopping condition if
		 * the change in LL is sufficiently small.
		 */
		double ll_old = 0.0d;
		boolean repeat = false;

		// initialize latent variable assignment and count matrices
		z_di = new int[D][];
		s_di = new int[D][];

		N_szd = new int[S][Z][D];
		N_swz = new int[S][W][Z];
		N_sw = new int[S][W];
		b_s  = new double[S];

		//N_z = new int[Z];
		N_d = new int[D];
		N_s = new int[S];
		N_sd = new int[S][D];
		a_sz = new double[S][Z];
		N_sz = new int[S][Z];

		P_s_w_z = new double[S][W][Z];
		P_s_z_d = new double[S][Z][D];

		System.out.println("Step 1: Initialize vectors.");
		for (int s = 0; s < S; s++)
			for (int z = 0; z < Z; z++){
				a_sz[s][z] = a/(S*Z);
			}

		for (int s = 0; s < S; s++){
			b_s[s] = b/S;
		}

		for (int d=0; d<D; d++) {  // document d

			int I = w_di[d].length;

			z_di[d] = new int[I];
			s_di[d] = new int[I];

			N_d[d] = I;

			int s = -1, z = -1;

			for (int i=0; i<w_di[d].length; i++) { 




				int w = w_di[d][i];		
				if (i == 0 || l_w[w_di[d][i-1]].equals("BR"))
				{
					s = (int) (S * Math.random());
					z = (int) (Z * Math.random());
				}

				s_di[d][i] = s;
				z_di[d][i] = z;				// Aadi: Initialize each word with a randomly gen. topic

				N_szd[s][z][d]++;				//Aadi: Increment the corresponding counters. N_zd, N_wz, N_z
				N_sd[s][d]++;
				N_swz[s][w][z]++;
				N_sw[s][w]++;
				N_sz[s][z]++;
				N_s[s]++;


			}

		}

		System.out.println("Step II: Iterate and revise");

		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {    // Aadi: For burn-in + samples number of iterations

			System.out.println("Iteration "+iteration+" begins.");

			for (int d=0; d<D; d++) { // document d

				int s = -1, z = -1;

				for (int i=0; i<w_di[d].length; ) { 

					int this_sent_i = i;
					int w = w_di[d][i];				// Aadi: Which word is this?
					z = z_di[d][i];				// Aadi: Which topic is it assigned to??
					s = s_di[d][i];				

					// Remove last values corresponding to s and z
					do
					{
						w = w_di[d][this_sent_i];

						N_sd[s][d]--;
						N_s[s]--;
						N_szd[s][z][d]--;
						N_sz[s][z]--;
						N_sw[s][w]--;


						N_swz[s][w][z]--;

						this_sent_i++;
					}while (this_sent_i < w_di[d].length && !l_w[w_di[d][this_sent_i-1]].equals("BR"));



					// position i
					// Aadi: Go over each word of all documents

					// Now, resample s



					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					double[] p = new double[S];
					double total = 0;

					for (s=0; s<S; s++) {

						p[s] = 0.0d;

						int sw_count = 0;

						int w_ = w_di[d][i];
						for (int pos = i; pos < this_sent_i; pos++)
						{
							sw_count = N_sw[s][w_];

						}

						
						// This could be wrong, Aditya.
						
						p[s] = ( sw_count + b_s[s])/(N_s[s] + b)* (N_sd[s][d] + g/S);




						total += p[s];
					}

					// resample Sentiment label
					double val = total * Math.random();

					s = 0; while ((val -= p[s]) > 0) s++;  // select a new topic

					// update latent variable and counts
					this_sent_i = i;
					do
					{
						w = w_di[d][this_sent_i];


						s_di[d][this_sent_i] = s;
						N_sd[s][d]++;
						N_s[s]++;
						N_sw[s][w]++;
						this_sent_i++;
					}while (this_sent_i < w_di[d].length && !l_w[w_di[d][this_sent_i -1]].equals("BR"));


					// Now, resample Z
					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					p = new double[Z];
					total = 0;


					for (z=0; z<Z; z++) {

						int swz_count = 0;

						for (int pos = i; pos < this_sent_i; pos++)
						{
							int w_ = w_di[d][pos];
							swz_count += N_swz[s][w_][z];

						}
						p[z] = ( (swz_count + b_s[s]/W)/(N_sz[s][z] + b_s[s]) ) * (N_szd[s][z][d] + a_sz[s][z]);
						total += p[z];

					}




					// resample 
					val = total * Math.random();
					if (val < 0)
						System.out.println("NEGATIVE WHAT!");
					z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

					// update latent variable and counts

					this_sent_i = i;
					
					do
					{
						w = w_di[d][this_sent_i];


						z_di[d][this_sent_i] = z;
						N_szd[s][z][d]++;
						N_sz[s][z]++;
						N_swz[s][w][z]++;
						this_sent_i++;

					}while (this_sent_i < w_di[d].length && !l_w[w_di[d][this_sent_i -1 ]].equals("BR"));


					/* Hop to the start of next sentence */
					i = this_sent_i + 1;

				}	

			}	


			// update parameter estimates
			if (iteration >= burnIn) {	

				a=0.0d;
				// Recompute a : the alphasum
				for (int s=0; s<S;s++) for (int z=0; z<Z; z++) a += a_sz[s][z]; 

				//Aadi: A sample is a complete configuration of probabilities at the end of an iter.
				for (int s=0;s<S;s++) for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_s_w_z[s][w][z] += (N_swz[s][w][z] + b_s[s]/W)/(N_sz[s][z] + b);
				for (int s=0;s<S;s++) for (int d=0; d<D; d++) for (int z=0; z<Z; z++) P_s_z_d[s][z][d] += (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a);


			}

			System.out.println("iteration: "+ iteration +", log-likelihood: "+ logLikelihood());
		}

		// normalize parameter estimates
		for (int s=0; s<S;s++) for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_s_w_z[s][w][z] /= samples;
		for (int s=0; s<S;s++) for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_s_z_d[s][z][d] /= samples;

		/* HE */
		System.out.println("Alpha values: ");
		for (int s=0; s<S;s++) 
			for (int i = 0; i < Z; i+=2){
				System.out.println("Z_"+s+"_"+i+" :"+a_sz[s][i]+"\tZ_"+s+"_"+(i+1)+" : "+a_sz[s][i+1]);
			}


		System.out.println("Saving parameters of model.");
		System.out.println("Final alpha values:");
		Training.printParameterSettings();
		for (int i=0; i<S; i++)
		{
			TopicModelUtils.saveMatrix(P_s_w_z[i],"P_"+i+"_w_z.data");
			TopicModelUtils.saveMatrix(P_s_z_d[i],"P_"+i+"_z_d.data");
		}
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");

		return P_s_w_z;		
	}	


	public double logLikelihood() {
		double ll = 0;
		for (int s=0; s<S; s++) { 
			for (int d=0; d<D; d++) { // document d
				for (int i=0; i<N_d[d]; i++) { // position i
					int z = z_di[d][i];
					int w = w_di[d][i];

					ll += Math.log( (N_sw[s][w] + g/S)/(N_s[s] + g) ); 
					ll += Math.log( (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a) );
					ll += Math.log( (N_swz[s][w][z] + b/S )/(N_sz[s][z] + b) );
				}
			}
		}

		return ll;
	}








	public double[] estimateP_d(){
		return estimateP_d(false);
	}

	public double[] estimateP_d(boolean smooth) { // Do some Jelinek-Mercer smoothing
		double lambda = 0.5;
		double uniform = 1/D;
		double[] p_d = new double[D];
		for (int d=0; d<D; d++)
			p_d[d] = smooth ? (lambda*((double)N_d[d]/N))+((1-lambda)*uniform) : (double)N_d[d]/N;
			return p_d;
	}


}
