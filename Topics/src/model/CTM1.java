package model;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv.CMAESOptimizer.Sigma;

import data.LabeledReviews;
import experiments.LDAIterator;
import experiments.Training;
import utils.Gamma;
import utils.LogisticNormal;
import utils.TopicModelUtils;


/*
 * Correlated Topic Model as given by David Blei.
 */
public class CTM1 {

	public static int Z; // number of topics
	public int W; // vocabulary
	public static int D; // number of documents
	public long N; // total word occurrences

	public double a;
	public double[] b_z;

	private double[][] b_wz;

	/* HE */
	public static double[] a_mu;	// a_mu[i] = mu for i'th topic
	public static double[][] a_sigma; // a_sigma[i][j] = correlation matrix

	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document

	public double[][] beta_dz; // beta_zd[z][d] = beta z'th topic in d'th document
	public double[][][] sigma_dzz; 
	public static int[][] N_zd; 
	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	public static int[] N_z;    // N_z[z] = count of z'th topic
	public int[] N_d;    // N_d[d] = length of document d


	public double[][] P_w_z;
	public double[][] P_z_d;

	public int num_samples;

	// private variables

	private static void update_mu_sigma()
	{
		for (int z=0;z<Z;z++)
		{
			a_mu[z] = N_z[z]/D;
		}
		
		for (int zi=0;zi<Z;zi++)
		{
			for (int zj=0;zj<Z;zj++)
			{
				double interm = 0;
				for (int d=0; d<D;d++)
				{
					interm += (N_zd[zi][d] - a_mu[zi]) *(N_zd[zj][d] - a_mu[zj]); 
				}


				a_sigma[zi][zj] = interm / (D-1);
			}

		}
	}
	
	/*
	 * loadbetapriors: Set beta priors based on word lists, if any.
	 * Same as in the case of LDA
	 */

	private static double[][] loadbetapriors(LabeledReviews lr, double val) {


		double [][] arr_bwz = new double[lr.W][Z];

		for (int w=0; w<lr.W; w++)
		{
			for (int z=0; z<Z; z++)
			{
				arr_bwz[w][z] = val;
			}

			if (lr.hm_sentiwordlist!=null && lr.hm_sentiwordlist.containsKey(lr.l_w[w]))
			{
				int polarity = lr.hm_sentiwordlist.get(lr.l_w[w]);
				if (polarity==1)
				{
					for (int z=0;z<Z/2;z++)
					{
						arr_bwz[w][z] = 2*val;
					}

					for (int z=Z/2;z<Z;z++)
					{
						arr_bwz[w][z] = 0;
					}
				}
				else
				{
					for (int z=Z/2;z<Z;z++)
					{
						arr_bwz[w][z] = 2*val;
					}

					for (int z=0;z<Z/2;z++)
					{
						arr_bwz[w][z] = 0;
					}
				}


			}

		}
		return arr_bwz;

	}

	public double[][] estimate(int[][] w_di, String[] l_w, int W, int D, long N, int Z, double alpha_prior, double beta_prior, int burnIn, int samples, int step, boolean hpestimate, LabeledReviews lr) {

		System.out.println("Estimation started!");
		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;

		/*
		 * Initialize beta priors with information from sentiment word list.
		 * a_mu is initialized to all 1's.
		 * a_sigma is an identity matrix.
		 */
		b_wz = loadbetapriors(lr, beta_prior);
		
		num_samples = 0;

		/*
		 * ll_old records ll value in last iteration. Repeat indicates stopping condition if
		 * the change in LL is sufficiently small.
		 */
		double ll_old = 0.0d;
		boolean repeat = false;

		// initialize latent variable assignment and count matrices
		z_di = new int[D][];
		b_z = new double[Z];
		beta_dz = new double[D][Z];

		N_zd = new int[Z][D];
		N_wz = new int[W][Z];
		N_z = new int[Z];
		N_d = new int[D];
		a_mu = new double[Z];
		a_sigma = new double[Z][Z];


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
			int z;

			z_di[d] = new int[I];
			N_d[d] = I;
			int polarity;

			for (int i=0; i<I; i++) { // position i

				if (lr.hm_sentiwordlist.containsKey(l_w[w_di[d][i]]))
				{
					polarity = lr.hm_sentiwordlist.get(l_w[w_di[d][i]]);
					if (polarity==1)
					{
						z = (0) + (int)(Math.random() * (((Z/2 -1) - (0)) + 1)); 

					}
					else
					{

						z = (Z/2) + (int)(Math.random()* (Z - 1 - (Z/2) + 1)); 
						//			System.out.println("Word "+ l_w[w_di[d][i]] + " is negative. gets initialized with topic : "+ z);
					}


				}
				else 
				{
					z =  (int)(Math.random() * (Z));
					//	System.out.println("Word "+ l_w[w_di[d][i]] + " is objective. gets initialized with topic : "+ z);
				}



				z_di[d][i] = z;	

				N_zd[z][d]++;
				N_wz[w_di[d][i]][z]++;
				N_z[z]++;
			}


		}

		/*
		 * Now initialize a_mu[z] and a_sigma[z]
		 */
	

		update_mu_sigma();

		/* Initialize beta */
		for (int d = 0; d < D; d++)
		{
			LogisticNormal ln = new LogisticNormal();
			beta_dz[d] = ln.averagedsample(a_mu, a_sigma, Z);
		}

		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {

			for (int d=0; d < D; d++) { // document d
				System.out.println("Iteration "+iteration+", document "+d);
				double cdf[] = new double[Z];
				double min_b[] = new double[Z];
				double max_b[] = new double[Z];
				/*
				 * Section 1: 
				 * Generate the topic-document distribution from a_mu, a_sigma
				 *
				 *
				 * Step 1: Using additional variable u_ij, find the margin to which existing betas
				 * can change.
				 */
				double sumexp = 0.0d;

				double C_t[] = new double[Z];

				for (int z=0;z<Z;z++)
				{
					
					sumexp = 0.0d;
					
					for (int z_=0;z_<Z;z_++)
					{
						
						sumexp += Math.exp(beta_dz[d][z_]);
					}

				
					
					/* Calculate C_t */
					C_t[z] = 1;
					for (int z_=0; z_<Z; z_++)
						if (z_!=z)
							C_t[z] += Math.exp(beta_dz[d][z_]);
					
					cdf[z] = Math.exp(beta_dz[d][z])/(1+sumexp);



					/*
					 * Generate u's for each word position
					 */

					for (int i=0; i<w_di[d].length;i++)
					{
						int curr_z = z_di[d][i];
						double min = 0, max =0;

						if (curr_z == z)
						{
							min = 0;
							max = cdf[z]; 
						}
						else
						{
							min = cdf[z];
							max = 1;
						}
						double range = max-min; 
						double u = (Math.random() * range) + min;

						if (curr_z == z)
						{
							double val2 = Math.log ((C_t[z] * u)/(1-u));

							if (max_b[z] < val2 || max_b[z] == 0)
								max_b[z] = val2;
						}
						else
						{
							double val2 = Math.log ((C_t[z] * u)/(1-u));
							if (min_b[z] > val2 || min_b[z] == 0)
								min_b[z] = val2;
						}
					}




				

				/* 
				 * max_b[z] and min_b[z] is the margin between which betas must change.
				 * 
				 * Step 2: Re-sample the beta
				 * 
				 * Selecting new beta uses a truncated normal distribution that lies between min_b[z] and max_b[z].
				 * The steps of sampling from such a distribution are:
				 * To update value of beta[z]
				 * 1) Get a uni-variate distribution over beta[z] | other betas.
				 * 2) Generate samples from this distribution. Eliminate ones outside min_b[z] 
				 * and max_b[z]. Repeat until 50 samples.
				 * 3) Randomly select one of the remaining.
				 */
				
					/* 1) Get uni-variate distribution */

					/* 1.1: Split mu into mu1 and mu2. mu1 is beta for this z */

					double[] mu1 = new double[1];
					mu1[0] = a_mu[z];

					RealMatrix r_mu1 = MatrixUtils.createColumnRealMatrix(mu1);

					double[] mu2 = new double[Z-1];


					int i=0;

					for (int z2=0;z2<Z;z2++)
					{
						if (z2!=z)
						{	
							mu2[i] = a_mu[z2];
							i++;
						}
					}

					RealMatrix r_mu2 = MatrixUtils.createColumnRealMatrix(mu2);

					double[][]s11 = new double[1][1];
					s11[0][0] = a_sigma[z][z];

					RealMatrix r_s11 = new Array2DRowRealMatrix(s11);

					double[][] s12 = new double[1][Z-1];
					i = 0;

					for (int z2=0;z2<Z;z2++)
					{
						if (z2!=z)
						{
							s12[0][i] = a_sigma[z][z2];
							i++;
						}

					}

					RealMatrix r_s12 = MatrixUtils.createRealMatrix(s12);


					double[][] s21 = new double[Z-1][1];
					i = 0;

					for (int z2=0;z2<Z;z2++)
					{
						if (z2!=z)
						{
							s21[i][0] = a_sigma[z2][z];
							i++;
						}

					}

					RealMatrix r_s21 = new Array2DRowRealMatrix(s21);


					double [][] s22 = new double[Z-1][Z-1];
					i = 0;
					int j = 0;
					for (int z2=0; z2<Z; z2++)
					{
						for (int z3=0; z3<Z; z3++)
						{
							if (z!=z2 && z!=z3)
							{
								s22[i][j] = a_sigma[z2][z3];
								j++;
							}

						}
						if(z!=z2)
						{
							i++;
							j=0;
						}
					}

					RealMatrix r_s22 = MatrixUtils.createRealMatrix(s22);

					/*
					 * Now the 'a' vector based on current values of betas
					 */

					double[] a = new double[Z-1];
					i = 0;

					for (int z2=0;z2<Z;z2++)
					{
						if(z2!=z)
						{
							a[i] = beta_dz[d][z2];
							i++;
						}
					}

					/*
					 * 1.2. now use equations that give you the univariate conditioned on others.
					 * These are terms in the equation for the new mu.
					 */
					RealMatrix r_s22i = new LUDecomposition(r_s22).getSolver().getInverse();
					RealMatrix r_a = MatrixUtils.createColumnRealMatrix(a);
					RealMatrix firstwo = r_s12.multiply(r_s22i);
					RealMatrix subtract =  r_a.subtract(r_mu2);
					RealMatrix finalone = firstwo.multiply(subtract);
					RealMatrix newMu = r_mu1.add(finalone); 

					finalone = firstwo.multiply(r_s21);
					RealMatrix newSigma = r_s11.subtract(finalone);


					double[][] mu = newMu.getData();
					double[][] sigma = newSigma.getData();


					NormalDistribution nd = new NormalDistribution(mu[0][0], sigma[0][0]);
					/* 2) Generate samples.
					 * 
					 * Now, generate 50 samples. Throw away things not in the bracket. 
					 */
					double sample = 0.0d;
					
						double current_val = nd.sample();

						/*
						 * Note that this min-max is correct. the min value is higher than the max value.
						 */
						while (!(current_val >= max_b[z] && current_val <= min_b[z]))
						{
							current_val = nd.sample();
							sample = current_val;
							
						}
					


					/* 3) Select one of the samples.
					 * Finally sample it! */

			
					
					sumexp -= beta_dz[d][z];
					beta_dz[d][z] = sample;
					sumexp += beta_dz[d][z];
				}
				/* 
				 * beta_dz[d][z] corresponds to beta for each z in this document d. We now
				 * sample each z.
				 */



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

						p[z] = ( (N_wz[w][z] + b_wz[w][z])/(N_z[z] + b_z[z]) ) * Math.exp(beta_dz[d][z]);
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


				//Aadi: A sample is a complete configuration of probabilities at the end of an iter.
				for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] += (N_wz[w][z] + b_wz[w][z])/(N_z[z] + b_z[z]);
				for (int d=0; d<D; d++) for (int z=0;z<Z; z++) P_z_d[z][d] = Math.exp(beta_dz[d][z]);


			}

			int hest_step = 0;
		
			step = 1;
			/* Now, reset the values of alpha and sigma */
			update_mu_sigma();

			if (iteration%step==0) System.out.println("iteration: "+ iteration +", log-likelihood: "+ logLikelihood());
		}

		// normalize parameter estimates
		for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_z_d[z][d] /= samples;

		/* Hyperparameter Estimation. Print final alpha values */



		System.out.println("Saving parameters of model.");
		System.out.println("Final alpha values:");

		TopicModelUtils.saveMatrix(P_w_z,"P_w_z.data");
		TopicModelUtils.saveMatrix(P_z_d,"P_z_d.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");

		return P_w_z;		
	}	


	public double logLikelihood() {
		double ll = 0;

		a = 0;
		for (int z=0; z<Z; z++){

			for (int w = 0; w < W; w++)
			{
				b_z[z] += b_wz[w][z];
			}

		}
		for (int d=0; d<D; d++) { 

			for (int z=0;z<Z;z++)
				ll += Math.log(beta_dz[d][z]);

			// document d
			for (int i=0; i<N_d[d]; i++) { // position i
				int z = z_di[d][i];
				int w = w_di[d][i];
				if (N_z[z] != 0)
				{
					ll += Math.log( (N_wz[w][z] + b_wz[w][z])/(N_z[z] + b_z[z]) ); 

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

	public static double leftToRightLikelihood(int[][] w_ij, double[][] p_w_z, double[] a_z, int R)
	{
		double ll = 0.0d;
		int D = w_ij.length;
		double t_b_dz[][] = new double[D][Z];
		
		for (int d =0 ; d < D; d++)
		{
			LogisticNormal ln = new LogisticNormal();
			t_b_dz[d] = ln.averagedsample(a_mu, a_sigma, Z);
			ll += leftToRightLikelihood(w_ij[d], p_w_z, a_z, t_b_dz[d], R);
		}

		return ll;
	}
	public static double leftToRightLikelihood(int[] w_j, double[][] p_w_z, double[] a_z, double[] t_b_dz, int R)
	{
		int n = w_j.length;
		double alphasum = 0.0d;

		for (int i =0; i < Z; i++)
		{
			alphasum += a_z[i];
		}

		double[] sump_i = new double[n];

		for (int r=0; r<R; r++) {
			int[] z_j = new int[n];
			int[] N_z = new int[Z];

			for (int i =0; i< n; i++)
			{

				// Skip unseen words in the test corpus
				if (w_j[i] == -1)
					continue;

				for (int j = 0; j < i; j++)
				{
					// Skip unseen words in the test corpus
					if (w_j[j] == -1)
						continue;

					int w = w_j[j];				
					int z = z_j[j];				

					// remove last value  			
					N_z[z]--;					

					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					double[] p = new double[Z];
					double total = 0;
					for (z=0; z<Z; z++) {
						p[z] = ( p_w_z[w][z] ) * (N_z[z] + a_z[z]);
						total += p[z];
					}

					// resample 
					double val = total * Math.random();
					z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

					// update latent variable and counts
					z_j[j] = z;

					N_z[z]++;   // update vars
				}

				int w = w_j[i];		
				for (int z = 0 ; z < Z ; z++) {		
					sump_i[i] += p_w_z[w][z] * (N_z[z]+a_z[z])/(i+alphasum); 
				}

				// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
				double[] p = new double[Z];
				double total = 0;
				int z;
				for (z=0; z<Z; z++) {
					p[z] = ( p_w_z[w][z] ) * (N_z[z] + a_z[z]);
					total += p[z];
				}

				// sample next z in vector 
				double val = total * Math.random();
				z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

				// update latent variable and counts
				z_j[i] = z;

				N_z[z]++;   // update vars

			}
		}


		//calculate likelihood
		double ll = 0d;
		for (int i=0; i<n; i++) {
			if (sump_i[i] != 0)
				ll += Math.log(sump_i[i]/R);
		}

		return ll;

	}

}