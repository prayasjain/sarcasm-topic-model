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
public class SLDA2PI {

	public static int Z; // number of topics
	public int W; // vocabulary
	public static int D; // number of documents
	public long N; // total word occurrences

	public double a; // topic smoothing hyperparameter \alpha
	public double b; // term smoothing hyperparameter \beta 

	/* HE */
	public static double[][] a_sz;	// a_z[i] = alpha for the i'th topic
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document. Same for alll words in a sentence.
	public static double[] g_s; // g_s[i] = gamma for i'th sentiment label
	public double[] a_s; // a_s[i] = Essentially the alphasum over all topics for i'th label

	public int[][] N_zd; // N_zd[z][d] = count of z'th topic in d'th document
	
	public static int[] N_d;    // N_d[d] = length of document d

	

	public int num_samples;
	public double g;
	/* New variables for this file */
	public static int S;		// No. of sentiment labels

	public int[][] s_di; // s_di[d][i] = sentiment assignment to i'th position in d'th document. Same for all words in a sentence.

	/* For gamma */
	
	/* For pi */
	public int[][] N_sd;	// N_sd[s][d] = count of s'th label in d'th document
	public int[] N_s;		// N_s[s] = count of s'th label

	/* For new phi */

	public int[][][] N_swz; // N_swz[s][w][z] = count of w'th word for z'th topic with s'th sentiment
	public int[][] N_sw; // N_sw[s][w] = count of w'th word with s'th sentiment


	/* For new theta */
	public static int[][][] N_szd; // N_szd[s][z][d] = count of s'th sentiment, z'th topic in d'th document
	public static int[][] N_sz; // N_sz[s][z] = count of s'th sentiment in z'th topic
	public int[][] N_wz;

	/* For new alpha */
	public double[][] b_sz; // b_sz[s] = beta for s'th sentiment
	public double[][][] b_swz; // b_sz[s][z]  = beta for s'th sentiment and t'th topic
	public double[][] b_sw;
	
	public double[][][] P_s_w_z;
	public double[][][] P_s_z_d;

	public static double[][] a_mu;
	public static double[][][][] a_sigma;
	
	/*
	 * To compare train and test 'user models'
	 */
	public static double tr_ll;
	public static double test_ll;
	public static int[] tr_bestSForZ;
	public static int[] test_bestSForZ;
	//??	public int[][] N_sz; // N_sz[s][z] = count of 
	
	private static double[] loadpriors(double g2) {
		// TODO Auto-generated method stub
		double [] arr_ad = new double[S];
		
		for (int z =0; z<S; z++)
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
	
	private static double[][][] loademotionpriors(int S, LabeledReviews lr, double b) {
		
		
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
				
				for (int z=0;z<Z;z++)
				{
					arr_bswz[polarity][w][z] = 2*b;
				}
					
				for (int p = 0; p < S; p++)
				for (int z=0;z<Z;z++)
				{
					if (p != polarity)
					arr_bswz[p][w][z] =0;
				}
				
				
			}
		}
		return arr_bswz;
		
	}

private static double[][][] loademotionpriors2(int S, LabeledReviews lr, double b) {
		
		
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
				
				for (int z=0;z<Z;z++)
				{
					arr_bswz[polarity][w][z] = 2*b;
				}
				
				int complementary_s = 0;
				switch(polarity)
				{
				case 0: complementary_s = 4;
						break;

				case 1: complementary_s = 5;
						break;

				case 2: complementary_s = 6;
						break;

				case 3: complementary_s = 7;
						break;

				case 4: complementary_s = 0;
						break;

				case 5: complementary_s = 1;
						break;

				case 6: complementary_s = 2;
						break;

				case 7: complementary_s = 3;
						break;
						
				}
				
				
				for (int z=0;z<Z;z++)
				{
					
					arr_bswz[complementary_s][w][z] =0;
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
		
		a_mu = new double[S][Z];
		a_sigma=new double[S][Z][S][Z];
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
		b_sw = new double[S][W];
		
		N_zd = new int[Z][D];

		//N_z = new int[Z];
		N_d = new int[D];
		N_s = new int[S];
		N_sd = new int[S][D];

		N_sz = new int[S][Z];
		
		
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
			
			if (w_di[d]==null)
				continue;
			
			int I = w_di[d].length;

			z_di[d] = new int[I];
			s_di[d] = new int[I];

			N_d[d] = I;

			int s = -1, z = -1;

			for (int i=0; i<w_di[d].length; i++) { 


				int w = w_di[d][i];		
				int polarity; 
				
				if (lr.hm_sentiwordlist.containsKey(l_w[w_di[d][i]]))
				{
					polarity = lr.hm_sentiwordlist.get(l_w[w_di[d][i]]);
					s = polarity;


				}
				else 
				{
					s =  (int)(Math.random() * (S));
					//	System.out.println("Word "+ l_w[w_di[d][i]] + " is objective. gets initialized with topic : "+ z);
				}
				
				
				
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
				
				if (w_di[d]==null)
					continue;
				
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
					
					a_s = new double[S];
					
					for (s=0; s<S; s++)
					{
						a_s[s] = 0;
						
						for(z=0; z<Z; z++)
							a_s[s] += a_sz[s][z];
					}
					
					for (s=0; s<S; s++) {

						for(z=0; z<Z; z++)
							p[s] += (N_szd[s][z][d] + a_sz[s][z])/(N_sd[s][d] + a_s[s]) * (N_swz[s][w][z] + b_swz[s][w][z])/(N_sz[s][z] + b_sz[s][z])* (N_sd[s][d] + g_s[s]);
						total += p[s];

						if (p[s] < 0)
							System.out.println("Exception for word pos " +i+" !");
					}

					// resample Sentiment label
					double val = total * Math.random();



					s = 0; while ((val -= p[s]) > 0) s++;  // select a new topic

					// update latent variable and counts

					

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
					 * The first part here is totally wrong. Incomplete, actually. This
					 * file was revised as SLDA3PI
					 */
					
					double betasum = 0.0d;
					for (int s = 0; s < S; s++)
					{
					
						for (int z=0; z < Z; z++)
						{
							
							betasum += b_sz[s][z];
						}
					}
					
					double denominator = 0.0d;
					
					
					for (int w=0; w<W; w++)
						for (int s = 0; s < S; s++)
						{
					
							double numerator = 0.0d;
						
							for (int z=0; z < Z; z++)
							{
								numerator += Gamma.digamma3(N_swz[s][w][z] + b_sw[s][w]);
							}
							
							numerator -= D*Gamma.digamma3(b_sw[s][w]);
						
							
							
							b_sw[s][w] *= numerator/denominator;
						}
				/*
				 * 
				 * Hyperparameter estimation. Update the gammas for each s.
				 */
				double gammasum = 0.0d;

				for (int i = 0; i < S; i++) gammasum += g_s[i];

				

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
		
			if (iteration%step==0) System.out.println("iteration: "+ iteration +", log-likelihood: "+ logLikelihood());
		
	}
	// normalize parameter estimates
	for (int s=0; s<S;s++) for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_s_w_z[s][w][z] /= samples;
	for (int s=0; s<S;s++) for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_s_z_d[s][z][d] /= samples;


	System.out.println("Gamma values: ");
	for (int s=0; s<S;s++) 
	{
		System.out.println("G_"+s+"_ :"+g_s[s]);
	}

	System.out.println("Final alpha values:");
	for (int s=0; s<S;s++) 
		for (int z = 0; z<Z; z++)
	{
		System.out.println("A_"+s+"_"+z+" :"+a_sz[s][z]);
	}
	
	
	System.out.println("Saving parameters of model.");
	
	
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

	g = 0;
	
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
	tr_ll = ll;
	return ll;
}

public static void getTopicCorrelation(){
	
	double hereN = 0.0d;
	
	for (int d = 0; d <D ; d++)
		hereN += N_d[d];
	
	
	System.out.print("Mean:");
	
	int counter = 0;
	
	for (int s = 0; s <S; s++)
	for (int z=0;z<Z;z++)
	{
		
		a_mu[s][z] = (double)(N_sz[s][z])/hereN;
		a_mu[s][z] = (double)Math.round(a_mu[s][z] * 10000) / 10000 ;
		System.out.print(a_mu[s][z]+" ");
	}
	
	System.out.println();
	
	System.out.println("Sigma");
	for (int si = 0; si < S; si++)
	{	for (int sj = 0; sj < S; sj++)
		{	
		for (int zi=0;zi<Z;zi++)
		{
			for (int zj=0;zj<Z;zj++)
			{
				double interm = 0;
				for (int d=0; d<D;d++)
				{
					interm += (N_szd[si][zi][d] - a_mu[si][zi]) *(N_szd[sj][zj][d] - a_mu[si][zj]); 
				}


			a_sigma[si][zi][sj][zj] = interm / (hereN-1);
			
			System.out.print((double)Math.round(a_sigma[si][zi][sj][zj] * 1000) / 1000 +" ");
		}
		
		
		System.out.println();
	}
	
		}
	}
	tr_bestSForZ = new int[Z];
	
	System.out.println("For the training corpus, these are the best estimates for the user's topic model");
	for (int z=0;z<Z;z++)
	{
		System.out.print(z+": ");
		int index = -1;
		int max = -1;
		for (int s=0;s<S;s++)
		{
			System.out.print(N_sz[s][z]+" ");
			if (N_sz[s][z] > max)
			{
				max = N_sz[s][z];
				index = s;
			}
			
		}
		
		tr_bestSForZ[z] = index;
		System.out.println();
	}
	
	calcMatches();
}

public static double leftToRightLikelihood(int[][] w_ij, double[][][] p_s_w_z, double[] g_s, double[][]a_sz, int R)
{
	
	double gammasum = 0.0d;
	double ll = 0d;
	int T_D = w_ij.length;
	for (int i =0; i < S; i++)
	{
		gammasum += g_s[i];
	}
	
	double[] a_s = new double[S];
	
	for (int s=0; s<S; s++)
		for (int z=0; z<Z; z++)
			a_s[s] += a_sz[s][z];
	
	test_bestSForZ = new int[Z];
	
	int[][] glN_sz = new int[S][Z];

	for (int d = 0 ; d < T_D; d++){
		if (w_ij[d] == null)
			continue;
		
		int n = w_ij[d].length;
		double[] sump_i = new double[n];
		
	for (int r=0; r<R; r++) {
		int[] z_j = new int[n];
		int[] N_z = new int[Z];
		int[] s_j = new int[n];
		int[] N_s = new int[S];
		int[][] N_sz = new int[S][Z];
		
		for (int i =0; i< n; i++)
		{
			
			// Skip unseen words in the test corpus
			if (w_ij[d][i] == -1)
				continue;
			
			for (int j = 0; j < i; j++)
			{
				// Skip unseen words in the test corpus
				if (w_ij[d][j] == -1)
					continue;
				
				int w = w_ij[d][j];				
				int z = z_j[j];				
				int s = s_j[j];
				// remove last value  			
				N_z[z]--;					
				N_s[s]--;
				N_sz[s][z]--;
				glN_sz[s][z]--;
				
				// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
				double[] p = new double[S];
				double total = 0;
				for (s=0; s<S; s++) {
					
					for(z=0; z<Z; z++)
						p[s] += ( p_s_w_z[s][w][z] ) * (N_s[s] + g_s[s]) * ((N_sz[s][z] + a_sz[s][z])/(N_s[s]+a_s[s]));
					
					total += p[s];
				}

				// resample 
				double val = total * Math.random();
				s = 0; while ((val -= p[s]) > 0) s++;  // select a new topic

				// update latent variable and counts
				s_j[j] = s;

				N_s[s]++;   // update vars
				
				
				p = new double[Z];
				total = 0;

				for (z=0; z<Z; z++) {
					p[z] = ( p_s_w_z[s][w][z] ) * (N_sz[s][z] + a_sz[s][z])/(N_s[s] + a_s[s]);
					total += p[z];
				}

				// resample 
				val = total * Math.random();
				if (val < 0)
					System.out.println("NEGATIVE WHAT!");
				
				z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

				z_j[j] = z;
				// update latent variable and counts
				N_sz[s][z]++;
				glN_sz[s][z]++;
				N_z[z]++;				
			}
			
			int w = w_ij[d][i];		
			for (int z = 0 ; z < Z ; z++) {	
				for (int s = 0; s < S; s++)
				sump_i[i] += p_s_w_z[s][w][z] * (N_sz[s][z]+a_sz[s][z])/(N_s[s]+a_s[s])* (N_s[s]+g_s[s])/(i+gammasum); 
			}
			
			// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
			double[] p = new double[S];
			double total = 0;
			int z, s;
			for (s=0; s<S; s++) {
				for(z=0; z<Z; z++)
					p[s] += ( p_s_w_z[s][w][z] ) * (N_s[s] + g_s[s]) * ((N_sz[s][z] + a_sz[s][z])/(N_s[s]+a_s[s]));
				
				total += p[s];
			}

			// resample 
			double val = total * Math.random();
			s = 0; while ((val -= p[s]) > 0) s++;  // select a new topic
			
			s_j[i] = s;
			
			N_s[s]++;   // update vars
			
			p = new double[Z];
			total = 0;

			for (z=0; z<Z; z++) {
				p[z] = ( p_s_w_z[s][w][z] ) * (N_sz[s][z] + a_sz[s][z])/(N_s[s] + a_s[s]);
				total += p[z];
			}

			// resample 
			val = total * Math.random();
			if (val < 0)
				System.out.println("NEGATIVE WHAT!");
			
			z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

			z_j[i] = z;
			
			// update latent variable and counts
			N_sz[s][z]++;
			glN_sz[s][z]++;
			N_z[z]++;			
			
		}
	}
	
			
	//calculate likelihood
	
	for (int i=0; i<n; i++) {
		if (sump_i[i] != 0)
			ll += Math.log(sump_i[i]/R);
	}
	
	
	}
	
	for (int z=0;z<Z;z++)
	{
		System.out.print(z+": ");
		int index = -1;
		int max = -1;
		for (int s=0;s<S;s++)
		{
			
			if (glN_sz[s][z] > max)
			{
				max = glN_sz[s][z];
				index = s;
			}
			
		}
		
		test_bestSForZ[z] = index;
		
		System.out.println();
	}
	test_ll = ll;
	
	
	return ll;
	
}



public double[] estimateP_d(){
	return estimateP_d(false);
}

/* This function evaluates the number of best matches between train and test user models */
public static void calcMatches()
{
	int matches = 0;
	
	for (int z=0;z<Z;z++)
	{
		if(tr_bestSForZ[z] == test_bestSForZ[z])
			matches++;
	}
	
	System.out.println("The match between train and test is for "+matches+" out of "+Z+" topics.");
	System.out.println("Train likelihood: "+tr_ll+", test likelihood: "+test_ll);
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
