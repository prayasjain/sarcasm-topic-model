package model;

import data.LabeledReviews;
import experiments.LDAIterator;
import experiments.Training;
import utils.Gamma;
import utils.TopicModelUtils;


/*
 * 24th October 2013.
 * 
 * This program implement Joint sentiment/aspect modeling.
 * It does NOT sample sentence level sentiment. However, it samples sentiment label first and then topic.
 * 
 * No hyperparameter estimation yet
 */
public class SLDA2P {

	public static int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public long N; // total word occurrences

	public double a; // topic smoothing hyperparameter \alpha
	public double b; // term smoothing hyperparameter \beta 

	/* HE */
	public double[][] a_sz;	// a_z[i] = alpha for the i'th topic
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document. Same for alll words in a sentence.
	public double[] g_s; // g_s[i] = gamma for i'th sentiment label
	public double[] a_s; // a_s[i] = Essentially the alphasum over all topics for i'th label

	public int[][] N_zd; // N_zd[z][d] = count of z'th topic in d'th document
	//	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	//	public int[] N_z;    // N_z[z] = count of z'th topic
	public int[] N_d;    // N_d[d] = length of document d

	//	public double[][] P_w_z;
	//	public double[][] P_z_d;

	public int num_samples;
	public double g;
	/* New variables for this file */
	public static int S;		// No. of sentiment labels

	public int[][] s_di; // s_di[d][i] = sentiment assignment to i'th position in d'th document. Same for all words in a sentence.

	/* For gamma */
	//public double g;	// sentiment smoothing hyperparameter \gamma

	/* For pi */
	public int[][] N_sd;	// N_sd[s][d] = count of s'th label in d'th document
	public int[] N_s;		// N_s[s] = count of s'th label

	/* For new phi */

	public int[][][] N_swz; // N_swz[s][w][z] = count of w'th word for z'th topic with s'th sentiment
	public int[][] N_sw; // N_sw[s][w] = count of w'th word with s'th sentiment


	/* For new theta */
	public int[][][] N_szd; // N_szd[s][z][d] = count of s'th sentiment, z'th topic in d'th document
	public int[][] N_sz; // N_sz[s][z] = count of s'th sentiment in z'th topic
	public int[][] N_wz;

	/* For new alpha */
	public double[][] b_sz; // b_sz[s] = beta for s'th sentiment
	public double[][][] b_swz; // b_sz[s][z]  = beta for s'th sentiment and t'th topic

	public double[][][] P_s_w_z;
	public double[][][] P_s_z_d;

	//??	public int[][] N_sz; // N_sz[s][z] = count of 
	
	private static double[] loadpriors(double g2) {
		// TODO Auto-generated method stub
		double [] arr_ad = new double[Z];
		
		for (int z =0; z<Z; z++)
			arr_ad[z] = g2;
		return arr_ad;
	}

	private static double[][] loadalphapriors(double g) {
		// TODO Auto-generated method stub
		double [][] arr_ad = new double[S][Z];
		for (int s=0; s<S; s++)
			for (int z =0; z<Z; z++)
				arr_ad[s][z] = g;
		
		return arr_ad;
	}

	private static double[][][] loadbetapriors(int S, LabeledReviews lr, double b) {
		
		
		double [][][] arr_bswz = new double[S][lr.W][Z];
		
		for (int w=0; w<lr.W; w++)
		{

			for(int s=0;s<S;s++)
				for (int z=0; z<Z; z++)
				{
					arr_bswz[s][w][z] = b;
				}
			
			if (lr.hm_sentiwordlist.containsKey(lr.l_w[w]))
			{
				int polarity = lr.hm_sentiwordlist.get(lr.l_w[w]);
				if (polarity==1)
				{
					for (int z=0;z<Z;z++)
					{
						arr_bswz[0][w][z] = 2*b;
					}
					
					for (int z=0;z<Z;z++)
					{
						arr_bswz[1][w][z] =0;
					}
				}
				else
				{
					for (int z=0;z<Z;z++)
					{
						arr_bswz[1][w][z] = 2*b;
					}
					
					for (int z=0;z<Z;z++)
					{
						arr_bswz[0][w][z] = 0;
					}
				}
			}
		}
		return arr_bswz;
		
	}

	public double[][][] estimate(int[][] w_di, String[] l_w,int W, int D, long N, int Z, int S, double a, double b, double g, int burnIn, int samples, int step, boolean hestimation, LabeledReviews lr) {

		this.w_di = w_di;
		this.S = S;
		
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		g_s = loadpriors(g);
		
		num_samples = 0;

		b_swz = loadbetapriors(S, lr, b);
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
		N_wz = new int[W][Z];
		b_sz = new double[S][Z];

		N_zd = new int[Z][D];

		//N_z = new int[Z];
		N_d = new int[D];
		N_s = new int[S];
		N_sd = new int[S][D];

		N_sz = new int[S][Z];
		g_s = new double[S];
		
		a_s = new double[S];
		
		P_s_w_z = new double[S][W][Z];
		P_s_z_d = new double[S][Z][D];

		
		this.a_sz = loadalphapriors(a);
		
		for (int s=0;s <S; s++)
		{
			for (int z=0; z<Z; z++)
			{
				for (int w=0;w<W;w++)
				{
					b_sz[s][z]+=b_swz[s][w][z];
				}
			}
		}
		
		
		System.out.println("Step 1: Initialize vectors.");

		

	



		for (int d=0; d<D; d++) {  // document d

			int I = w_di[d].length;

			z_di[d] = new int[I];
			s_di[d] = new int[I];

			N_d[d] = I;

			int s = -1, z = -1;

			for (int i=0; i<w_di[d].length; i++) { 


				int w = w_di[d][i];		
				s = (int) (S * Math.random());
				z = (int) (Z * Math.random());

				s_di[d][i] = s;
				z_di[d][i] = z;				// Aadi: Initialize each word with a randomly gen. topic

				N_szd[s][z][d]++;				//Aadi: Increment the corresponding counters. N_zd, N_wz, N_z
				N_sd[s][d]++;
				N_swz[s][w][z]++;
				N_sw[s][w]++;
				N_sz[s][z]++;
				N_s[s]++;
				N_zd[z][d]++;
				N_wz[w][z]++;
			}

		}

		System.out.println("Step II: Iterate and revise");

		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {    // Aadi: For burn-in + samples number of iterations

			for (int d=0; d<D; d++) { // document d

				int s = -1, z = -1;

				for (int i=0; i<w_di[d].length; i++) { 


					int w = w_di[d][i];				// Aadi: Which word is this?
					z = z_di[d][i];				// Aadi: Which topic is it assigned to??
					s = s_di[d][i];				

					// Remove last values corresponding to s and z

					N_sd[s][d]--;
					N_s[s]--;
					N_szd[s][z][d]--;
					N_sz[s][z]--;
					N_sw[s][w]--;
					N_swz[s][w][z]--;
					N_wz[w][z]--;
					N_zd[z][d]--;

					// position i
					// Aadi: Go over each word of all documents

					// Now, resample s
					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					double[] p = new double[S];
					double total = 0;

					for (s=0; s<S; s++)
					{
						for(z=0; z<Z; z++)
							a_s[s] += a_sz[s][z];
					}
					
					for (s=0; s<S; s++) {

						for(z=0; z<Z; z++)
							p[s] += (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a_s[s]) * (N_swz[s][w][z] + b_swz[s][w][z])/(N_sw[s][w] + b_sz[s][z])* (N_sd[s][d] + g_s[s]);
						total += p[s];

						if (p[s] < 0)
							System.out.println("Exception for word pos " +i+" !");
					}

					// resample Sentiment label
					double val = total * Math.random();



					s = 0; while ((val -= p[s]) > 0) s++;  // select a new topic

					// update latent variable and counts

					w = w_di[d][i];

					s_di[d][i] = s;
					N_sd[s][d]++;
					N_s[s]++;
					N_sw[s][w]++;

					// Now, resample Z
					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					p = new double[Z];
					total = 0;

					for (z=0; z<Z; z++) {
						p[z] = ( (N_swz[s][w][z] + b_swz[s][w][z])/(N_sz[s][z] + b_sz[s][z]) ) * (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a_s[s]);
						total += p[z];
					}

					// resample 
					val = total * Math.random();
					if (val < 0)
						System.out.println("NEGATIVE WHAT!");
					z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

					// update latent variable and counts
					w = w_di[d][i];
					z_di[d][i] = z;
					N_szd[s][z][d]++;
					N_sz[s][z]++;
					N_swz[s][w][z]++;
					N_wz[w][z]++;
					N_zd[z][d]++;
				}	

			}	


			// update parameter estimates
			if (iteration >= burnIn) {	



				//Aadi: A sample is a complete configuration of probabilities at the end of an iter.
				for (int s=0;s<S;s++) for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_s_w_z[s][w][z] += (N_swz[s][w][z] + b_swz[s][w][z])/(N_sz[s][z] + b_sz[s][z]);
				for (int s=0;s<S;s++) for (int d=0; d<D; d++) for (int z=0; z<Z; z++) P_s_z_d[s][z][d] += (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a_s[s]);
			}

			if(hestimation)
			{
				do{
			

				/*
				 * 
				 * Hyperparameter estimation. Update the gammas for each s.
				 */
				double gammasum = 0.0d;

				for (int i = 0; i < S; i++) gammasum += g_s[i];

				double denominator = 0.0d;

				for (int d = 0; d <D; d++)
					denominator += Gamma.digamma(N_d[d] + gammasum) ;

				denominator -= D * Gamma.digamma(gammasum);

				gammasum = 0.0d;

				for (int s= 0; s <S; s++)
				{
					double numerator = 0.0d;

					for (int d = 0; d <D; d++)
						numerator += Gamma.digamma(N_sd[s][d] + g_s[s]) ;

					numerator -= D * Gamma.digamma(g_s[s]);

					g_s[s] = g_s[s] * (numerator / denominator);
					gammasum += g_s[s];
				}

				double bigalphasum = 0.0d;
				for (int i = 0; i <S; i++)
					a_s[i] = 0;
				
				for (int s = 0; s < S; s++){
					/*
					 * 
					 * Hyperparameter estimation. Update the gammas for each s.
					 */
					double alphasum = 0.0d;

					for (int z = 0; z < Z; z++) alphasum += a_sz[s][z];

					denominator = 0.0d;

					for (int d = 0; d <D; d++)
						denominator += Gamma.digamma(N_d[d] + alphasum) ;

					denominator -= D * Gamma.digamma(alphasum);

					alphasum = 0.0d;

					for (int z= 0; z <Z; z++)
					{
						double numerator = 0.0d;

						for (int d = 0; d <D; d++)
							numerator += Gamma.digamma(N_szd[s][z][d] + a_sz[s][z]) ;

						numerator -= D * Gamma.digamma(a_sz[s][z]);

						a_sz[s][z] = a_sz[s][z] * (numerator / denominator);
						alphasum += a_sz[s][z];
						a_s[s] += a_sz[s][z];
					}
					
					bigalphasum+= alphasum;
				}

				/*
				 * Estimate log-likelihood. Stop when log likelihood differs by a small value
				 */

				double ll = 0.0d;
				double likelihood = 0.0d;

				for (int s = 0; s < S; s++)
				{
					likelihood *= Gamma.gamma(N_s[s]+g_s[s])/Gamma.gamma(g_s[s]);

				}

				likelihood *= Gamma.gamma(N + gammasum) / Gamma.gamma(gammasum);

				for (int s = 0; s < S; s++)
				{	
					for (int z = 0; z < Z; z++)
					{
						likelihood *= Gamma.gamma(N_sz[s][z]+a_sz[s][z])/Gamma.gamma(a_sz[s][z]);

					}

				likelihood *= Gamma.gamma(N_s[s] + a_s[s]) / Gamma.gamma(a_s[s]);
				}
				ll = Math.log(likelihood);

				repeat = (Math.abs(ll - ll_old) > 0.0001d) ? true : false;

				ll_old = ll;
			}while(repeat);
			}
		
			System.out.println("iteration: "+ iteration +", log-likelihood: "+ logLikelihood());
		
	}
	// normalize parameter estimates
	for (int s=0; s<S;s++) for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_s_w_z[s][w][z] /= samples;
	for (int s=0; s<S;s++) for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_s_z_d[s][z][d] /= samples;


	System.out.println("Gamma values: ");
	for (int s=0; s<S;s++) 
	{
		System.out.println("G_"+s+"_ :"+g_s[s]);
	}

	for (int s=0; s<S;s++) 
		for (int z = 0; z<Z; z++)
	{
		System.out.println("A_"+s+"_"+z+" :"+a_sz[s][z]);
	}
	System.out.println("Saving parameters of model.");
	System.out.println("Final alpha values:");
	
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

	for (int s=0;s<S; s++)
		g+=g_s[s];
	
	for (int d=0; d<D; d++) { // document d
		for (int i=0; i<N_d[d]; i++) { // position i
			int z = z_di[d][i];
			int w = w_di[d][i];
			int s = s_di[d][i];


			ll += Math.log( (N_szd[s][z][d] + a_sz[s][z])/(N_d[d] + a_s[s]) );
			ll += Math.log( (N_sd[s][d] + g_s[s])/(N_d[d] + g) );
			ll += Math.log( (N_swz[s][w][z] + b_swz[s][w][z])/(N_sz[s][z] + b_sz[s][z]) );
		}
	}
	LDAIterator i = new LDAIterator();
	i.ll = ll;

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
