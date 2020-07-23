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
public class LDA2PI {

	public static int Z; // number of topics
	public int W; // vocabulary
	public static int D; // number of documents
	private static long N; // total word occurrences
	private static int L ; // labels
	private static int S ; // number of sentiment

	private static long[][] N_z_l ; // Count document level topic and label
	private static long[] N_l ; // Count document level label
	private static long[][][] N_s_zl ; //Count word level sentiment topic and label
	private static long[][] N_zlw ; // Count word level topic and label
	private int[][] is_di; // Switch for dth document ith word
	private int[][] sentiment ; //Sentiment of dth document ith word

	public int[] label ; // label of dth document
	public int[] topic ; //topic of dth document


	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document

	public int num_samples;

	public void estimate(int[] label, int[][] w_di, String[] l_w, int W, int D, long N, int Z, int L, int S , double alpha_prior, double beta_prior, int burnIn, int samples, int step, boolean hpestimate, LabeledReviews lr) {
		System.out.println("Estimation started!");
		this.w_di = w_di;
		this.label= label ;
		this.Z = Z;
		this.L = L ;
		this.S = S ;
		this.W = W;
		this.D = D;
		this.setN(N);

//		b_wz = loadbetapriors(lr, beta_prior);
//		a_z = loadalphapriors(alpha_prior);

		num_samples=0 ;
		N_z_l= new long[Z][L] ;
		N_l = new long[L] ;
		N_s_zl = new long[S][Z][L];
		N_zlw = new long[Z][L] ;
		is_di = new int[D][500] ;
		sentiment = new int[D][500] ;
		topic = new int[D] ;

		for(int d=0;d<D;d++){
			topic[d]= Math.random()*Z ;
		}
		for(int d=0;d<D;d++){
			for(int i=0;i<w_di[d].length;i++){
				is_di[d][i] = Math.random()*2 ;
				if(is_di[d][i]==1)
					sentiment[d][i]= Math.random()*S ;
				else
					sentiment[d][i] =-1 ;
			}
		}


		for(int d=0;d<D;d++){
			N_z_l[topic[d]][label[d]]++ ;
			N_l[label[d]]++ ; 
		}
		for(int d=0;d<D;d++){
			long z =topic[d] ;
			long l =label[d] ;
			for(int i=0;i<w_di[d].length;i++){
				long w = w_di[d][i];				
				int is = is_di[d][i];				
				int s = sentiment[d][i];

				if(is==1){
					N_s_zl[s][z][l]++ ;
				} 						

			}
			N_zlw[z][l]+=w_di.length; 

		}



		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples*step; iteration++) {
			
			for (int d=0; d<D; d++) { // document d
				
				if (w_di[d] == null)
					continue;
				l=label[d];
				z=topic[d] ;
				//Estimate probability of z/l
				double[] p = new double[Z] ;
				N_z_l[z][l]--;
				N_l[l]-- ;
				N_zlw[z][l]-=w_di[d].length ;
				for(int i=0;i<w_di[d].length;i++){
					if(is_di[d][i]==1)
						N_s_zl[sentiment[d][i]][z][l]-- ;

				}



				for(int zi=0;zi<Z;zi++){

					p[zi] = N_zi_l[zi][l]/N_l[l] ;

				}
				double val = Math.random() ;
				z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic
				N_z_l[z][l]++;
				N_l[l]++ ;
				N_zlw[z][l]+=w_di[d].length ;
				for(int i=0;i<w_di[d].length;i++){
					if(is_di[d][i]==1)
						N_s_zl[sentiment[d][i]][z][l]++ ;

				}


				for (int i=0; i<w_di[d].length; i++) { // position i
					// Aadi: Go over each word of all documents
					
					long w = w_di[d][i];				// Aadi: Which word is this?
					int is = is_di[d][i];				//whether it is a topic word or sentiment word
					int s = sentiment[d][i] ; 							//sentiment of word
					

					if(is==1){
						//calculate p(s/Z,l,w) /propto p(s/Z,l) * p(w/s)
						N_s_zl[s][z][l]-- ;
						N_zlw[z][l]-- ;

						double[] p = new double[S];

						double p_w_s; // ? p(w/s)
						double p_s_zl; 
						double total =0 ;
						for(senti=0;senti<S;senti++){
							p_w_s=1 ;
							p_s_zl = N_s_zl[senti][z][l]/N_zlw[z][l] ;
							p[senti] = p_w_s*p_s_zl ;
							total+=p[senti] ;
						}
						for(senti=0;senti<S;senti++)
							p[senti]/=total ;
						double val = Math.random();
 						s=0;while ((val -= p[s]) > 0) s++;
 						N_s_zl[s][z][l]++ ;
 						N_zlw[z][l]++ ;
					}

					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)

				}	
			}	

			// update parameter estimates
			if (iteration >= burnIn) {	

					//fill
			}

			int hest_step = 0;
			
			if (hpestimate)
			{
				//fill
			}


		}
		
	}





	private static double[] loadalphapriors(double a) {
		// TODO Auto-generated method stub
		double [] arr_ad = new double[Z];
		
		for (int z =0; z<Z; z++)
			arr_ad[z] = a;
		return arr_ad;
	}

private static double[][] loaduniformbetapriors(LabeledReviews lr, double val) {
		
		
		double [][] arr_bwz = new double[lr.W][Z];
		
		for (int w=0; w<lr.W; w++)
		{

			
			for (int z=0; z<Z; z++)
			{
				arr_bwz[w][z] = val;
			}
		}
	return arr_bwz;
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
					for (int z=0;z<Z/3;z++)
					{
						arr_bwz[w][z] = 2*val;
					}
					
					for (int z=Z/3;z<Z;z++)
					{
						arr_bwz[w][z] = 0;
					}
				}
				else
				{
					for (int z=Z/3;z<2*Z/3;z++)
					{
						arr_bwz[w][z] = 2*val;
					}
					
					for (int z=0;z<Z/3;z++)
					{
						arr_bwz[w][z] = 0;
					}
					for (int z=2*Z/3;z<Z;z++)
					{
						arr_bwz[w][z] = 0;
					}
				}


			}
			
			
			
		}
		return arr_bwz;
		
	}

private static double[][] loademotionpriors(LabeledReviews lr, double val) {
		
		
		double [][] arr_bwz = new double[lr.W][Z];
		
		for (int w=0; w<lr.W; w++)
		{

			
			for (int z=0; z<Z; z++)
			{
				arr_bwz[w][z] = val;
			}
			
			if (lr.hm_sentiwordlist.containsKey(lr.l_w[w]))
			{
				int this_z = lr.hm_sentiwordlist.get(lr.l_w[w]);
				
				if (this_z < 10)
				{
					
					int reset = this_z;
					
					for (int z = 0; z < Z; z++)
					{
						if (z == reset)
						{
							arr_bwz[w][reset] = 2*val;
							reset = z + 8 ;
						}
						else
						{
							arr_bwz[w][z] = 0;
						}
					}
					
					
				}
				else
				{
					
				}
			}
		}
		return arr_bwz;
		
	}

/*
 * Considers the axial structure of emotions. joy = ~sadness, anticipation = ~surprise, etc.
 */

private static double[][] loademotionpriors2(LabeledReviews lr, double val) {
	
	
	double [][] arr_bwz = new double[lr.W][Z];
	
	for (int w=0; w<lr.W; w++)
	{

		
		for (int z=0; z<Z; z++)
		{
			arr_bwz[w][z] = val;
		}
		
		if (lr.hm_sentiwordlist.containsKey(lr.l_w[w]))
		{
			int this_z = lr.hm_sentiwordlist.get(lr.l_w[w]);
			
			if (this_z < 10)
			{
				
					arr_bwz[w][this_z] = 2*val;
					
				
				
				int complementary_z = 0;
				switch(this_z)
				{
				case 0: complementary_z = 4;
						break;

				case 1: complementary_z = 5;
						break;

				case 2: complementary_z = 6;
						break;

				case 3: complementary_z = 7;
						break;

				case 4: complementary_z = 0;
						break;

				case 5: complementary_z = 1;
						break;

				case 6: complementary_z = 2;
						break;

				case 7: complementary_z = 3;
						break;
						
				}
				
				arr_bwz[w][complementary_z] = 0;
			}
			else
			{
				
			}
		}
	}
	return arr_bwz;
	
}
		


	public static void getTopicCorrelation(){
		
		double hereN = 0.0d;
		
		for (int d = 0; d <D ; d++)
			hereN += N_d[d];
		
		System.out.print("Mean:");
		

		for (int z=0;z<Z;z++)
		{
			
			a_mu[z] = (double)(N_z[z])/hereN;
			a_mu[z] = (double)Math.round(a_mu[z] * 10) / 10 ;
			System.out.print(a_mu[z]+"");
		}
		
		System.out.println();
		
		System.out.println("Sigma");
		
		for (int zi=0;zi<Z;zi++)
		{
			for (int zj=0;zj<Z;zj++)
			{
				double interm = 0;
				for (int d=0; d<D;d++)
				{
					interm += (N_zd[zi][d] - a_mu[zi]) *(N_zd[zj][d] - a_mu[zj]); 
				}


				a_sigma[zi][zj] = interm / (hereN-1);
				
				System.out.print((double)Math.round(a_sigma[zi][zj] * 1000) / 1000 +" ");
			}
			
			
			System.out.println();
		}
		
		
	}
	
	
	public double logLikelihood() {
		double ll = 0;
		
		a = 0;
		for (int z=0; z<Z; z++){
			
			a += a_z[z];
			b_z[z] = 0;
			
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
			p_d[d] = smooth ? (lambda*((double)N_d[d]/getN()))+((1-lambda)*uniform) : (double)N_d[d]/getN();
			return p_d;
	}

	public static double leftToRightLikelihood(int[][] w_ij, double[][] p_w_z, double[] a_z, int R)
	{
		double ll = 0.0d;
		int D = w_ij.length;
		
		for (int i =0 ; i < D; i ++) { if(w_ij[i]!=null && w_ij[i].length>0)	ll += leftToRightLikelihood(w_ij[i], p_w_z, a_z, R);
		}
		return ll;
	}
	public static double leftToRightLikelihood(int[] w_j, double[][] p_w_z, double[] a_z, int R)
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

	public static long getN() {
		return N;
	}

	public static void setN(long n) {
		N = n;
	}
	
}
