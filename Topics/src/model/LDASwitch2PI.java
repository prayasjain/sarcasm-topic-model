package model;

import data.LabeledReviews;
import experiments.Training;
import utils.Gamma;
import utils.TopicModelUtils;

/* 
 * Validated on 24th October
 */
/*
 * Includes hyperparameter estimation
 * 
 * Implements switching variable. Passes in s as number of distributions to choose from.
 * 
 * In progress on 4th November. Must be checked for completeness.
 */

public class LDASwitch2PI {

	public static int[] Z; // number of topics in each distribution. Z.length = total number of alternate distributions
	public int W; // vocabulary
	public int D; // number of documents
	public long N; // total word occurrences
	public static int S;
	
	public double[] a_s; // topic smoothing hyperparameter \alpha
	public double[][][] b_swz; // term smoothing hyperparameter \beta 
	public double[] g_s; // gamma for hyperparameter
	public double[] b_s;
	
	/* HE */
	public double[][] a_sz;	// a_sz[s][z] = alpha for the z'th topic of s'th distribution
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document
	public int[][] s_di; // s_di[d][i] = distribution identifier assignment to i'th position in d'th document
	
	public int[] N_s;
	public int[][][] N_szd; // N_szd[s][z][d] = count of z'th topic of s'th distribution in d'th document
	public int[][][] N_wsz; // N_wsz[w][s][z] = count of w'th word for z'th topic of s'th distribution
	public int[][] N_sz;    // N_sz[s][z] = count of z'th topic of r'th distribution
	public int[] N_d;    // N_d[d] = length of document d
	public int[][] N_sd; // N_sd[s][d] = count of s'th distribution in d'th document
	public int[][] N_sw;
	
	public double[][][] P_s_w_z;
	public double[][][] P_s_z_d;

	public int num_samples;

	// private variables
	private double[] b_on_W; // b_on_W[r] = b_on_w for distribution r
	private static double[] loadpriors(double g2) {
		// TODO Auto-generated method stub
		double [] arr_ad = new double[S];
		
		for (int z =0; z<S; z++)
			arr_ad[z] = g2;
		return arr_ad;
	}

	private static double[][] loadalphapriors(double g) {
		// TODO Auto-generated method stub
		double [][] arr_ad = new double[S][];
		for (int s=0; s<S; s++)
		{	
			arr_ad[s] = new double[Z[s]];
			for (int z =0; z<Z[s]; z++)
				arr_ad[s][z] = g;
		}
		return arr_ad;
	}

	private static double[][][] loadbetapriors(int S, LabeledReviews lr, double b) {
		
		
		double [][][] arr_bswz = new double[S][lr.W][];
		
		for (int w=0; w<lr.W; w++)
		{

			for(int s=0;s<S;s++)
			{
				arr_bswz[s][w] = new double[Z[s]];
				for (int z=0; z<Z[s]; z++)
				{
					
					arr_bswz[s][w][z] = b;
				}
			}
			
			if (lr.hm_sentiwordlist.containsKey(lr.l_w[w]))
			{
				int polarity = lr.hm_sentiwordlist.get(lr.l_w[w]);
				if (polarity==1)
				{
					int s = 1;
					int z = 0;
					arr_bswz[s][w][z] = 2*b;
					arr_bswz[s][w][1] = 0;
					s = 0;
					
					for (z=0;z<Z[s];z++)
					{
						arr_bswz[s][w][z] = 0;
					}
				}
				else
				{
					int s = 1;
					int z = 1;
					arr_bswz[s][w][z] = 2*b;
					arr_bswz[s][w][0] = 0;
					
					s = 0;
					
					for (z=0;z<Z[s];z++)
					{
						arr_bswz[s][w][z] = 0;
					}
				}
			}
		}
		return arr_bswz;
		
	}


	public double[][][] estimate(int[][] w_di, int W, int D, long N, int S, int[] Z, double a, double b, double g, int burnIn, int samples, int step, boolean hestimation, LabeledReviews lr) {

		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		
		//this.b_s = b;
		this.S = S;
		
		num_samples = 0;
		
		b_s = new double[S];
		b_swz = new double[S][][];
		g_s = new double[S];
		a_sz = new double[S][];
		
		b_swz = loadbetapriors(S, lr, b);
		
		
		
		a_sz = loadalphapriors(a);
		
		g_s = loadpriors(g);
		
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
		
		N_szd = new int[S][][];
		N_wsz = new int[W][S][];
		N_sz = new int[S][];
		N_d = new int[D];
		N_sd = new int[S][D];
		
		N_sw = new int[S][W];
		N_s = new int[S];
		
		P_s_w_z = new double[S][W][];
		P_s_z_d = new double[S][][];

		for (int i = 0; i < S; i++)
		{
			int curr_z = Z[i];

			for (int j = 0; j < W; j++)
			{
				N_wsz[j][i] = new int[curr_z];
				P_s_w_z[i][j] = new double[curr_z];
			}
			
			N_szd[i] = new int[curr_z][D];
			N_sz[i] = new int[curr_z];
			a_sz[i] = new double[curr_z];
			P_s_z_d[i] = new double[curr_z][D];
		}

		for (int d=0; d<D; d++) {  // document d

			int I = w_di[d].length;

			z_di[d] = new int[I];
			s_di[d] = new int[I];
			
			N_d[d] = I;
			for (int i=0; i<I; i++) { // position i
				int s = (int) (S * Math.random());
				int z = (int) (Z[s] * Math.random());
				s_di[d][i] = s;
				z_di[d][i] = z;				// Aadi: Initialize each word with a randomly gen. topic
				N_szd[s][z][d]++;				//Aadi: Increment the corresponding counters. N_zd, N_wz, N_z
				N_wsz[w_di[d][i]][s][z]++;
				N_sz[s][z]++;
				N_sd[s][d]++;
				N_sw[s][w_di[d][i]]++;
				N_s[s]++;
			}
		}

		int[][] b_sz;
		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {    // Aadi: For burn-in + samples number of iterations
			for (int d=0; d<D; d++) { // document d
				for (int i=0; i<w_di[d].length; i++) { // position i
					// Aadi: Go over each word of all documents
					int w = w_di[d][i];				// Aadi: Which word is this?
					int z = z_di[d][i];				// Aadi: Which topic is it assigned to??
					int s = s_di[d][i];
					
					// remove last value  			
					N_szd[s][z][d]--;					// Aadi: Reduce counts corresponding to this word and this topic
					N_wsz[w][s][z]--;
					N_sz[s][z]--;
					N_sd[s][d]--;
					N_sw[s][w]--;
					N_s[s]--;
					
					// Sample r
					double[] p = new double[S];
					double total = 0;
					p = new double[S];
					
					for (s = 0; s < S; s++)
					{
						for (int z_ = 0; z_< Z[s]; z_++)
							p[s] += (N_szd[s][z_][d] + a_sz[s][z_])/(N_sd[s][d] + a_s[s]) * (N_wsz[w][s][z_] + b_sz[s][z_])/(N_sz[s][z_] + b_s[s])* (N_sd[s][d] + g_s[s]);
						total += p[s];
					}
					
					double val = total * Math.random();
					s = 0; while ((val -= p[s]) > 0) s++; // select a new r 
					s_di[d][i] = s;
					
					
					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					p = new double[Z[s]];
					total = 0;
					for (z=0; z<Z[s]; z++) {
						p[z] = ( (N_wsz[w][s][z] + b_sz[s][z]/W)/(N_sz[s][z] + b_sz[s][z]) ) * (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a_s[s]);
						total += p[z];
					}

					// resample 
					val = total * Math.random();
					z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

					// update latent variable and counts
					z_di[d][i] = z;

					N_szd[s][z][d]++;   // update vars
					N_wsz[w][s][z]++;
					N_sz[s][z]++;
					N_sw[s][w]++;
					N_sd[s][d]++;
					N_s[s]++;

				}	
			}	

			// update parameter estimates
			if (iteration >= burnIn) {	

				
				//Aadi: A sample is a complete configuration of probabilities at the end of an iter.
				for (int s=0;s<S;s++) for (int w=0; w<W; w++) for (int z=0; z<Z[s]; z++) P_s_w_z[s][w][z] += (N_wsz[w][s][z] + b_sz[s][z])/(N_sz[s][z] + b_s[s]);
				for (int s=0;s<S;s++) for (int d=0; d<D; d++) for (int z=0; z<Z[s]; z++) P_s_z_d[s][z][d] += (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a_s[s]);

				

			}
			
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

					for (int z = 0; z < Z[s]; z++) alphasum += a_sz[s][z];

					denominator = 0.0d;

					for (int d = 0; d <D; d++)
						denominator += Gamma.digamma(N_d[d] + alphasum) ;

					denominator -= D * Gamma.digamma(alphasum);

					alphasum = 0.0d;

					for (int z= 0; z <Z[s]; z++)
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
					for (int z = 0; z < Z[s]; z++)
					{
						likelihood *= Gamma.gamma(N_sz[s][z]+a_sz[s][z])/Gamma.gamma(a_sz[s][z]);

					}

				likelihood *= Gamma.gamma(N_s[s] + a_s[s]) / Gamma.gamma(a_s[s]);
				}
				ll = Math.log(likelihood);

				repeat = (Math.abs(ll - ll_old) > 0.0001d) ? true : false;

				ll_old = ll;
			}while(repeat);

			g= 0.0d;
			for (int i = 0; i < S; i++)
				g += g_s[i];

			if (iteration%step==0) System.out.println("iteration: "+ iteration +", log-likelihood: "+ logLikelihood());
		}

		// normalize parameter estimates
		for (int s=0; s<S;s++) for (int w=0; w<W; w++) for (int z=0; z<Z[s]; z++) P_s_w_z[s][w][z] /= samples;
		for (int s=0; s<S;s++) for (int d=0; d<D; d++) for (int z=0; z<Z[s]; z++)	P_s_z_d[s][z][d] /= samples;

		/* Hyperparameter Estimation. Print final alpha values */
		System.out.println("Alpha values: ");
		for (int s=0; s<S;s++) 
			for (int i = 0; i < Z[s]; i++){
				System.out.println("Z_"+s+"_"+i+" :"+b_sz[s][i]);
			}
		System.out.println("Saving parameters of model.");
		System.out.println("Final alpha values:");
		Training.printParameterSettings();
		for (int i=0; i<S; i++)
		{
			TopicModelUtils.saveMatrix(P_s_w_z[i],"P_"+i+"_w_z.data");
			System.out.println("11111");
			TopicModelUtils.saveMatrix(P_s_z_d[i],"P_"+i+"_z_d.data");
		}
		

		return P_s_w_z;		
	}	


	public double logLikelihood() {
		double ll = 0;
		double gsum = 0.0d;
		
		for (int s=0; s<S; s++)
			gsum += g_s[s];
		
	
			for (int d=0; d<D; d++) { // document d
				for (int i=0; i<N_d[d]; i++) { // position i
					int z = z_di[d][i];
					int w = w_di[d][i];
					int s = s_di[d][i];
					
					ll += Math.log( (N_sd[s][d] + g_s[s])/(N_d[d] + gsum) ); // pi
					ll += Math.log( (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a_s[s]) ); // theta
					ll += Math.log( (N_wsz[w][s][z] + b_sz[s][z])/(N_sz[s][z] + b_s[s]) );
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
