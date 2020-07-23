package model;

import utils.TopicModelUtils;



public class OTM1 {
	// Aadi: Here, topic assignments will be resampled. In addition to words, you also have opinion label information.
	
	public int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public int O; // number of opinion labels
	public long N; // total word occurrences
	
	public double a; // topic smoothing hyperparameter \alpha
	public double b; // term smoothing hyperparamter \beta 
	
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document
	public int[] o_d;  // o_d[d] = observed opinion label for d'th document
	
	public int[][] N_zd;    // N_zd[z][d] = count of z'th topic in d'th document
	public int[][][] N_wzo; // N_wzo[w][z][o] = count of w'th word for z'th topic and o'th opinion label
	public int[][] N_zo;    // N_zo[z][o] = count of z'th topic and o'th opinion label
		
	// private variables
	private double b_on_W;
	private double a_on_Z;
	
	
	public void estimate(int[][] w_di, int[] o_d, int W, int D, long N, int Z, double a, double b, int iterations, int step) {
		
		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		this.a = a;
		this.b = b;
		
		b_on_W = b/W;
		a_on_Z = a/Z;
		
		
		// initialize latent variable assignment and count matrices
		z_di = new int[D][];
		N_zd = new int[Z][D];
		N_wzo = new int[W][Z][O];
		N_zo = new int[Z][O];
		
		for (int d=0; d<D; d++) {  // document d
			int I = w_di[d].length;
			z_di[d] = new int[I];
			int o = o_d[d];
			for (int i=0; i<I; i++) { // position i
				int z = (int) (Z * Math.random());
				z_di[d][i] = z;
				N_zd[z][d]++;
				N_wzo[w_di[d][i]][z][o]++;
				N_zo[z][o]++;
			}
		}
		
		// perform Gibbs sampling
		for (int iteration=0; iteration<iterations; iteration++) {
			for (int d=0; d<D; d++) { // document d

				int o = o_d[d];
				
				for (int i=0; i<w_di[d].length; i++) { // position i
					
					int w = w_di[d][i];
					int z = z_di[d][i];
					
					// remove last value  
					N_zd[z][d]--;
					N_wzo[w][z][o]--;
					N_zo[z][o]--;
					
					//Aadi: We have additional observed information here in the form of opinion labels. Therefore, this 
					
					
					// calculate distribution p(z|w,o,d) = p(z,w|o,d)/p(w|o,d) /propto p(z,w|o,d) = p(w|z,o)p(z|d)
					double[] p = new double[Z];
					double total = 0;
					for (int k=0; k<Z; k++) {
						p[k] = ( (N_wzo[w][k][o] + b_on_W)/(N_zo[k][o] + b) ) * (N_zd[k][d] + a_on_Z);
						total += p[k];
					}
					
					// resample 
					double val = total * Math.random();
					z = 0; while ((val -= p[z]) > 0) z++;
					
					// update latent variable and counts
					z_di[d][i] = z;
					
					N_zd[z][d]++;
					N_wzo[w][z][o]++;
					N_zo[z][o]++;
					
				}	
			}	
			if (iteration%step==0) System.out.println("iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.save3dMatrix(estimateP_w_zo(),"P_w_zo.data");
		TopicModelUtils.saveMatrix(estimateP_z_d(),"P_z_d.data");
		
	}	
		
	
	public double logLikelihood() {
		double ll = 0;
		for (int d=0; d<D; d++) { // document d
			int N_d = w_di[d].length;
			int o = o_d[d];
			for (int i=0; i<N_d; i++) { // position i
				int z = z_di[d][i];
				int w = w_di[d][i];
				ll += Math.log( (N_wzo[w][z][o] + b_on_W)/(N_zo[z][o] + b) ); 
				ll += Math.log( (N_zd[z][d] + a_on_Z)/(N_d + a) );
			}
		}
		return ll;
	}
	
	
	public double[][][] estimateP_w_zo() {
		double[][][] p_w_zo = new double[W][Z][O];
		for (int w=0; w<W; w++) 
			for (int z=0; z<Z; z++) 
				for (int o=0; o<O; o++) 
						p_w_zo[w][z][o] = (N_wzo[w][z][o] + b_on_W)/(N_zo[z][o] + b);
		return p_w_zo;
	}
	
		
	public double[][] estimateP_z_d() {
		double[][] p_z_d = new double[Z][D];
		for (int d=0; d<D; d++) {
			int N_d = w_di[d].length;
			for (int z=0; z<Z; z++)
				p_z_d[z][d] = (N_zd[z][d] + a_on_Z)/(N_d + a);
		}	
		return p_z_d;
	}
	
	

}
