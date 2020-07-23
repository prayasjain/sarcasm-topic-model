package model;

import utils.TopicModelUtils;



public class OTM2 {
	//Aadi: Here, opinion assignments and topic assignments, will both be resampled.
	
	public int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public int O; // number of opinion labels
	public long N; // total word occurrences
	
	public double a; // topic smoothing hyperparameter \alpha
	public double b; // term smoothing hyperparamter \beta 
	public double g; // term smoothing hyperparamter \gamma
	
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document
	public int[][] o_di; // o_di[d][i] = opinion assignment for i'th position in d'th document
	
	public int[][] N_zd;    // N_zd[z][d] = count of z'th topic in d'th document
	public int[][][] N_wzo; // N_wzo[w][z][o] = count of w'th word for z'th topic and o'th opinion label
	public int[][] N_zo;    // N_zo[z][o] = count of z'th topic and o'th opinion label
	public int[][] N_od; 	// N_od[o][d] = count of the o'th opinion label in d'th document
	
	// private variables
	private double b_on_W;
	private double a_on_Z;
	private double g_on_O;
	
	
	public void estimate(int[][] w_di, int W, int D, long N, int Z, double a, double b, double g, int iterations, int step) {
		
		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		this.a = a;
		this.b = b;
		this.g = g;
		
		b_on_W = b/W;
		a_on_Z = a/Z;
		g_on_O = g/O;
		
		// initialize latent variable assignment and count matrices
		z_di = new int[D][];
		o_di = new int[D][];
		N_zd = new int[Z][D];
		N_wzo = new int[W][Z][O];
		N_zo = new int[Z][O];
		N_od = new int[O][D];
		
		for (int d=0; d<D; d++) {  // document d
			int I = w_di[d].length;
			z_di[d] = new int[I];
			o_di[d] = new int[I];
			for (int i=0; i<I; i++) { // position i
				int w = w_di[d][i];
				int z = (int) (Z * Math.random());
				int o = (int) (O * Math.random());
				z_di[d][i] = z;
				o_di[d][i] = o;
				N_zd[z][d]++;
				N_wzo[w][z][o]++;
				N_zo[z][o]++;
				N_od[o][d]++;
			}
		}
		
		// perform Gibbs sampling
		for (int iteration=0; iteration<iterations; iteration++) {
			for (int d=0; d<D; d++) { // document d
				for (int i=0; i<w_di[d].length; i++) { // position i
					
					int w = w_di[d][i];
					int z = z_di[d][i];
					int o = o_di[d][i];

					// first resample o
					
					// remove o's counts
					N_wzo[w][z][o]--;
					N_zo[z][o]--;
					N_od[o][d]--;
					
					// calculate distribution p(o|w,z,d) = p(o,w|z,d)/p(w|z,d) \propto p(o,w|z,d) = p(w|o,z)p(o|d) 
					double[] p = new double[O];
					double total = 0;
					for (int k=0; k<O; k++) {
						p[k] = ((N_wzo[w][z][k] + b_on_W)/(N_zo[z][k] + b)) * (N_od[k][d] + g_on_O);
						total += p[k];
					}
					// resample 
					double val = total * Math.random();
					o = 0; while ((val -= p[o]) > 0) o++;
					
					// update o's counts
					//N_wzo[w][z][o]++;
					//N_zo[z][o]++;
					N_od[o][d]++;

					// now sample z
					
					// remove last value  
					//N_wzo[w][z][o]--;
					//N_zo[z][o]--;
					N_zd[z][d]--;
					
					// calculate distribution p(z|w,o,d) = p(z,w|o,d)/p(w|o,d) /propto p(z,w|o,d) = p(w|z,o)p(z|d)
					p = new double[Z];
					total = 0;
					for (int k=0; k<Z; k++) {
						p[k] = ( (N_wzo[w][k][o] + b_on_W)/(N_zo[k][o] + b) ) * (N_zd[k][d] + a_on_Z);
						total += p[k];
					}
					// resample 
					val = total * Math.random();
					z = 0; while ((val -= p[z]) > 0) z++;
					
					// update latent variable and counts
					z_di[d][i] = z;
					
					N_wzo[w][z][o]++;
					N_zo[z][o]++;
					N_zd[z][d]++;
					
				}	
			}	
			if (iteration%step==0) System.out.println("iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.save3dMatrix(estimateP_w_zo(),"P_w_zo.data");
		TopicModelUtils.saveMatrix(estimateP_z_d(),"P_z_d.data");
		TopicModelUtils.saveMatrix(estimateP_o_d(),"P_o_d.data");
		
	}	
		
	
	public double logLikelihood() {
		double ll = 0;
		for (int d=0; d<D; d++) { // document d
			int N_d = w_di[d].length;
			for (int i=0; i<N_d; i++) { // position i
				int z = z_di[d][i];
				int o = o_di[d][i];
				int w = w_di[d][i];
				ll += Math.log( (N_wzo[w][z][o] + b_on_W)/(N_zo[z][o] + b) ); 
				ll += Math.log( (N_zd[z][d] + a_on_Z)/(N_d + a) );
				ll += Math.log( (N_od[o][d] + g_on_O)/(N_d + g) );		
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
	
	
	public double[][] estimateP_o_d() {
		double[][] p_o_d = new double[O][D];
		for (int d=0; d<D; d++) {
			int N_d = w_di[d].length;
			for (int o=0; o<O; o++)
				p_o_d[o][d] = (N_od[o][d] + g_on_O)/(N_d + g);
		}	
		return p_o_d;
	}
	
	
}
