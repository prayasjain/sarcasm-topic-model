package model;

import data.LabeledReviews;
import experiments.LDAIterator;
import experiments.Training;
import utils.Gamma;
import utils.TopicModelUtils;

/* 
 * Validated on 24th October
 */
/*
 * Includes hyperparameter estimation
 */
public class LDA2P {

	public static int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public long N; // total word occurrences

	public double a;
	public double[] b_z;

	private double[][] b_wz;

	/* HE */
	public double[] a_z;	// a_z[i] = alpha for the i'th topic
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document

	public int[][] N_zd; // N_zd[z][d] = count of z'th topic in d'th document
	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	public int[] N_z;    // N_z[z] = count of z'th topic
	public int[] N_d;    // N_d[d] = length of document d


	public double[][] P_w_z;
	public double[][] P_z_d;

	public int num_samples;

	// private variables

	private static double[] loadalphapriors(double a) {
		// TODO Auto-generated method stub
		double [] arr_ad = new double[Z];
		
		for (int z =0; z<Z; z++)
			arr_ad[z] = a;
		return arr_ad;
	}

	private static double[][] loadbetapriors(LabeledReviews lr, double val) {
		
		
		double [][] arr_bwz = new double[lr.W][Z];
		
		for (int w=0; w<lr.W; w++)
		{

			
			for (int z=0; z<Z; z++)
			{
				arr_bwz[w][z] = val;
			}
			
			if (lr.hm_sentiwordlist.containsKey(lr.l_w[w]))
			{
				int polarity = lr.hm_sentiwordlist.get(lr.l_w[w]);
				if (polarity==1)
				{
					for (int z=0;z<Z/2;z++)
					{
						arr_bwz[w][z] = 2*val;
					}
				}
				else
				{
					for (int z=Z/2;z<Z;z++)
					{
						arr_bwz[w][z] = 2*val;
					}
				}


			}
			
			
			
		}
		return arr_bwz;
		
	}

	public double[][] estimate(int[][] w_di, int W, int D, long N, int Z, double alpha_prior, double beta_prior, int burnIn, int samples, int step, boolean hpestimate, LabeledReviews lr) {
		System.out.println("Estimation started!");
		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;

		b_wz = loadbetapriors(lr, beta_prior);
		a_z = loadalphapriors(a);
		
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
		b_z = new double[Z];
		N_zd = new int[Z][D];
		N_wz = new int[W][Z];
		N_z = new int[Z];
		N_d = new int[D];
		a_z = new double[Z];
		b_wz = new double[W][Z];

		P_w_z = new double[W][Z];
		P_z_d = new double[Z][D];


		



		


		for (int z=0; z<Z; z++)
		{
			for (int w=0;w<W;w++)
			{
				b_z[z]+=b_wz[w][z];
			}
		}

		for (int d=0; d<D; d++) {  // document d

			int I = w_di[d].length;

			z_di[d] = new int[I];
			N_d[d] = I;
			for (int i=0; i<I; i++) { // position i
				int z = (int) (Z * Math.random());
				z_di[d][i] = z;				// Aadi: Initialize each word with a randomly gen. topic
				N_zd[z][d]++;				//Aadi: Increment the corresponding counters. N_zd, N_wz, N_z
				N_wz[w_di[d][i]][z]++;
				N_z[z]++;
			}
		}

		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {    // Aadi: For burn-in + samples number of iterations
			for (int d=0; d<D; d++) { // document d
				for (int i=0; i<w_di[d].length; i++) { // position i
					// Aadi: Go over each word of all documents
					int w = w_di[d][i];				// Aadi: Which word is this?
					int z = z_di[d][i];				// Aadi: Which topic is it assigned to??

					// remove last value  			
					N_zd[z][d]--;					// Aadi: Reduce counts corresponding to this word and this topic
					N_wz[w][z]--;
					N_z[z]--;

					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					double[] p = new double[Z];


					double total = 0;
					for (z=0; z<Z; z++) {

						p[z] = ( (N_wz[w][z] + b_wz[w][z])/(N_z[z] + b_z[z]) ) * (N_zd[z][d] + a_z[z]);
						total += p[z];
					}

					// resample 
					double val = total * Math.random();
					z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

					// update latent variable and counts
					z_di[d][i] = z;

					N_zd[z][d]++;   // update vars
					N_wz[w][z]++;
					N_z[z]++;

				}	
			}	

			// update parameter estimates
			if (iteration >= burnIn) {	

				a=0.0d;
				double b=0.0d;

				// Recompute a : the alphasum
				for (int z=0; z<Z; z++){
					a += a_z[z];

					for (int w = 0; w < W; w++)
					{
						b_z[z] += b_wz[w][z];
					}

				}

				//Aadi: A sample is a complete configuration of probabilities at the end of an iter.
				for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] += (N_wz[w][z] + b_wz[w][z])/(N_z[z] + b_z[z]);
				for (int d=0; d<D; d++) for (int z=0; z<Z; z++) P_z_d[z][d] += (N_zd[z][d] + a_z[z])/(N_d[d] + a);


			}

			if (hpestimate)
			{
			do{

				/*
				 * 
				 * Hyperparameter estimation. Update the alphas for each z.
				 */
				double alphasum = 0.0d;

				for (int i = 0; i < Z; i++) alphasum += a_z[i];

				double denominator = 0.0d;

				for (int d = 0; d <D; d++)
					denominator += Gamma.digamma3(N_d[d] + alphasum) ;

				denominator -= D * Gamma.digamma3(alphasum);

				alphasum = 0.0d;

				for (int z = 0; z <Z; z++)
				{
					double numerator = 0.0d;

					for (int d = 0; d <D; d++)
						numerator += Gamma.digamma3(N_zd[z][d] + a_z[z]) ;
					double old_az = a_z[z];
					numerator -= D * Gamma.digamma3(a_z[z]);

					a_z[z] = a_z[z] * (numerator / denominator);

					
				//	if (Double.isNaN(a_z[z])) {
			//			System.err.println("Log likelihood is a NaN. Possibly a division by zero error");
						
				//	}
					alphasum += a_z[z];
				}


				/*
				 * Estimate log-likelihood. Stop when log likelihood differs by a small value
				 */

				double ll = 0.0d;
				double likelihood = 0.0d;

				for (int z = 0; z < Z; z++)
				{
					likelihood *= Gamma.gamma(N_z[z]+a_z[z])/Gamma.gamma(a_z[z]);

				}

				likelihood *= Gamma.gamma(N + alphasum) / Gamma.gamma(alphasum);

				ll = Math.log(likelihood);

				repeat = (Math.abs(ll - ll_old) > 0.0001d) ? true : false;

				ll_old = ll;
			}while(repeat);

			}


			if (iteration%step==0) System.out.println("iteration: "+ iteration +", log-likelihood: "+ logLikelihood());
		}

		// normalize parameter estimates
		for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_z_d[z][d] /= samples;

		/* Hyperparameter Estimation. Print final alpha values */
		System.out.println("Alpha values: ");
		for (int i = 0; i < Z; i++)
			System.out.println("Z_"+i+" :"+a_z[i]);

		System.out.println("Saving parameters of model.");
		System.out.println("Final alpha values:");

		TopicModelUtils.saveMatrix(P_w_z,"P_w_z.data");
		TopicModelUtils.saveMatrix(P_z_d,"P_z_d.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");

		return P_w_z;		
	}	


	public double logLikelihood() {
		double ll = 0;
		for (int z=0; z<Z; z++){
			a += a_z[z];

			for (int w = 0; w < W; w++)
			{
				b_z[z] += b_wz[w][z];
			}

		}
		for (int d=0; d<D; d++) { // document d
			for (int i=0; i<N_d[d]; i++) { // position i
				int z = z_di[d][i];
				int w = w_di[d][i];
				if (N_z[z] != 0)
				{
					ll += Math.log( (N_wz[w][z] + b_wz[w][z])/(N_z[z] + b_z[z]) ); 

					//if (Double.isNaN(ll)) {
					//	System.err.println("Log likelihood is a NaN. Possibly a division by zero error");

					//}

					ll += Math.log( (N_zd[z][d] + a_z[z])/(N_d[d] + a) );

				//	if (Double.isNaN(ll)) {
				//		System.err.println("Log likelihood is a NaN. Possibly a division by zero error");
				//	}
				}
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
