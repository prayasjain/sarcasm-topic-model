package model;

import utils.TopicModelUtils;

/*
 * 
 * No no no no no. This is wrong!
 * 
 *  A word is either a sentiment word or a topic word. This is similar to Arjun Mukherjee's
 *  use of switch variables
 *  to determine if a word is derived from a topic distribution or a comment-expression distribution.
 */


public class LDAWithSwitch1 {
	//Aadi: Assignments of words to topics are the only latent vars here
	public int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public long N; // total word occurrences
	
	public double a1; // topic smoothing hyperparameter \alpha
	public double b1; // term smoothing hyperparameter \beta 
	public double a2; // topic smoothing hyperparameter \alpha for sentiment category
	public double b2; // term smoothing hyperparameter \beta  for sentiment category
	
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] t_di; // t_di[d][i] = topic assignment, if any, to i'th position in d'th document
	public int[][] s_di; // s_di[d][i] = sentiment category assignment, if any, to i'th position in d'th document 
	
	public int[][] N_zd; // N_zd[z][d] = count of z'th topic in d'th document
	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	public int[] N_z;    // N_z[z] = count of z'th topic
	public int[] N_s;	// N_s[s] = count of s'th sentiment category
	
	public int[][] N_sd; // N_sd[s][d] = count of s'th sentiment category in d'th document
	public int[][] N_ws; // N_ws[w][s] = count of w'th word for s'th sentiment category
	
	public int[] N_d;    // N_d[d] = length of document d
	 
	public double[][] P_w_z;
	public double[][] P_z_d;
	
	public double[][] P_w_s;
	public double[][] P_s_d;
	
	public int num_samples;
	
	// private variables
	private double a1_on_Z;
	private double b1_on_W;
	private double a2_on_S;
	private double b2_on_W;
	private double bern_p;
	
	public double[][] estimate(int[][] w_di, int W, int D, long N, int Z, int S, double a1, double b1, double a2, double b2, double bern_p, int burnIn, int samples, int step) {
						
		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		this.a1 = a1;
		this.b1 = b1;
		this.a2 = a2;
		this.b2 = b2;
		num_samples = 0;
		b1_on_W = b1/W;
		a1_on_Z = a1/Z;
		b2_on_W = b2/W;
		a2_on_S = a2/S;
		
		
		// initialize latent variable assignment and count matrices
		t_di = new int[D][];
		N_zd = new int[Z][D];
		N_wz = new int[W][Z];
		N_z = new int[Z];
		N_d = new int[D];
		
		s_di = new int[D][];
		N_sd = new int[S][D];
		N_ws = new int[W][S];
		N_s = new int[S];
		
		P_w_z = new double[W][Z];
		P_z_d = new double[Z][D];
		
		for (int d=0; d<D; d++) {  // document d
			
			int I = w_di[d].length;
			
			t_di[d] = new int[I];
			s_di[d] = new int[I];
			
			N_d[d] = I;
			
			for (int i=0; i<I; i++) { // position i
				
				// Sample switching variable at random.
				int r = (int) (2 * Math.random());
				
				int z, s;
				
				if (r == 0)
				{
					z = (int) (Z * Math.random());
					
					s_di[d][i] = -1;
					t_di[d][i] = z;				// Aadi: Initialize each word with a randomly gen. topic
					N_zd[z][d]++;				//Aadi: Increment the corresponding counters. N_zd, N_wz, N_z
					N_wz[w_di[d][i]][z]++;
					N_z[z]++;
				}
				else
				{
					s = (int) (S * Math.random());
					
					t_di[d][i] = -1;
					s_di[d][i] = s;				// Aadi: Initialize each word with a randomly gen. sentiment category
					N_sd[s][d]++;				//Aadi: Increment the corresponding counters. N_sd, N_ws, N_s
					N_ws[w_di[d][i]][s]++;
					N_s[s]++;
				}
				
				
			}
		}
		
		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {    
			for (int d=0; d<D; d++) { // document d
				for (int i=0; i<w_di[d].length; i++) { // position i
														
					int w = w_di[d][i];				
					int z = t_di[d][i];				
					int s = s_di[d][i];
					int r;							// The switch variable
					if (z != -1)
					{
						// remove last value  			
						N_zd[z][d]--;
						N_wz[w][z]--;
						N_z[z]--;
					}
					else if (s != -1)
					{
						N_sd[s][d]--;
						N_ws[w][s]--;
						N_s[s]--;
					}
					
					// Sample r. This time based on Bernoulli prior "p"
					double[] p = new double[2];
					p[0] = bern_p;
					p[1] = 1 - bern_p;
					double total = 1;
					
					double val = total * Math.random();
					r = 0;
					while ((val -= p[r]) > 0) r++;
					
					// If r = 0, sample a topic. If r = 1, sample a sentiment category.
					if (r == 0)
					{
						// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
						p = new double[Z];
						total = 0;
						for (z=0; z<Z; z++) {
							p[z] = ( (N_wz[w][z] + b1_on_W)/(N_z[z] + b1) ) * (N_zd[z][d] + a1_on_Z);
							total += p[z];
						}
					
						// resample 
						val = total * Math.random();
						z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic
						
						// update latent variable and counts
						t_di[d][i] = z;
						s_di[d][i] = -1;
						
						N_zd[z][d]++;   // update vars
						N_wz[w][z]++;
						N_z[z]++;
					
					}	
					else
					{
						// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
						p = new double[S];
						total = 0;
						for (s=0; s<S; s++) {
							p[s] = ( (N_ws[w][s] + b2_on_W)/(N_s[s] + b2) ) * (N_sd[s][d] + a2_on_S);
							total += p[s];
						}
					
						// resample 
						val = total * Math.random();
						s = 0; while ((val -= p[s]) > 0) s++;  // select a new topic
						
						// update latent variable and counts
						s_di[d][i] = s;
						t_di[d][i] = -1;
						
						N_sd[s][d]++;   // update vars
						N_ws[w][s]++;
						N_s[s]++;
					}
				}
			}	

			// update parameter estimates
			if (iteration >= burnIn) {	//Aadi: A sample is a complete configuration of probabilities at the end of an iter.
				for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] += (N_wz[w][z] + b1_on_W)/(N_z[z] + b1);
				for (int d=0; d<D; d++) for (int z=0; z<Z; z++) P_z_d[z][d] += (N_zd[z][d] + a1_on_Z)/(N_d[d] + a1);
				
				for (int w=0; w<W; w++) for (int s=0; s<S; s++) P_w_s[w][s] += (N_wz[w][s] + b2_on_W)/(N_z[s] + b2);
				for (int d=0; d<D; d++) for (int s=0; s<S; s++) P_s_d[s][d] += (N_zd[s][d] + a2_on_S)/(N_d[d] + a2);
			}
			
			if (iteration%step==0) System.out.println("iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}
		
		// normalize parameter estimates
		for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_z_d[z][d] /= samples;
		
		for (int w=0; w<W; w++) for (int s=0; s<S; S++)	P_w_s[w][s] /= samples;
		for (int d=0; d<D; d++) for (int s=0; s<S; S++)	P_s_d[s][d] /= samples;
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.saveMatrix(P_w_z,"P_w_z.data");
		TopicModelUtils.saveMatrix(P_z_d,"P_z_d.data");
		TopicModelUtils.saveMatrix(P_w_s,"P_w_s.data");
		TopicModelUtils.saveMatrix(P_s_d,"P_s_d.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");
		
		return P_w_z;		
	}	
		
	
	public double logLikelihood() {
		double ll = 0;
		for (int d=0; d<D; d++) { // document d
			for (int i=0; i<N_d[d]; i++) { // position i
				int z = t_di[d][i];
				int s = s_di[d][i];
				int w = w_di[d][i];
				
				if (s == -1){
					ll += Math.log( (N_wz[w][z] + b1_on_W)/(N_z[z] + b1) ); 
					ll += Math.log( (N_zd[z][d] + a1_on_Z)/(N_d[d] + a1) );
				}
				else {
					ll += Math.log( (N_ws[w][s] + b2_on_W)/(N_s[s] + b2) ); 
					ll += Math.log( (N_sd[s][d] + a2_on_S)/(N_d[d] + a2) );
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
