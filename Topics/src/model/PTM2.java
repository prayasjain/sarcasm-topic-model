package model;

import utils.TopicModelUtils;

public class PTM2 {
	
	public int Z;  // number of topics
	public int W;  // vocabulary
	public int U;  // number of users
	public int D;  // number of documents (clicked urls)
	public long N; // total word occurrences
	
	public double a;  // topic smoothing hyperparameter for users
	public double bd; // document smoothing hyperparameter for topics 
	public double bw; // term smoothing hyperparamter for topics (\beta) 
	
	public int[][][] w_uij; // w_uij[u][i][j] = j'th word in the i'th query by the u'th user
	public int[][][] z_uij; // z_uij[u][i][j] = topic assignment to j'th position in the i'th query by the u'th user
	public int[][] d_ui;    // d_ui[u][i] = document for i'th query by the u'th user
	
	public int[][] N_zu; // N_zu[z][u] = count of z'th topic for the u'th user
	public int[][] N_zd; // N_zd[z][d] = count of z'th topic for the d'th document
	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	public int[] N_u;    // N_u[u] = count of u'th user
	public int[] N_d;    // N_d[d] = count of d'th document
	public int[] N_z;    // N_z[z] = count of z'th topic
		
	// private variables
	private double bw_on_W;
	private double bd_on_D;
	private double a_on_Z;
	
	// parameter estimates
	public double[][] P_w_z;   
	public double[][] P_d_z; 
	public double[][] P_z_u; 
	
	public double[][] estimate(int[][][] w_uij, int[][] d_ui, int W, int D, long N, int Z, double a, double bd, double bw, int burnin, int samples, int step) {
		
		this.w_uij = w_uij;
		this.d_ui = d_ui;
		this.W = W;
		this.U = w_uij.length;
		this.D = D;
		this.N = N;
		this.Z = Z;
		this.a = a;
		this.bd = bd;
		this.bw = bw;

		bw_on_W = bw/W;
		bd_on_D = bd/D;
		a_on_Z = a/Z;
		
		// initialise param matrices
		P_w_z = new double[W][Z];
		P_d_z= new double[D][Z];
		P_z_u = new double[Z][U];
		
		// initialize latent variable assignment and count matrices
		z_uij = new int[U][][];
		N_zu = new int[Z][U];
		N_zd = new int[Z][D];
		N_wz = new int[W][Z];
		N_u = new int[U];
		N_d = new int[D];
		N_z = new int[Z];
		for (int u=0; u<U; u++) {  // user u
			int I = w_uij[u].length;
			z_uij[u] = new int[I][];
			for (int i=0; i<I; i++) { // query i
				int J = w_uij[u][i].length;
				z_uij[u][i] = new int[J];
				int d = d_ui[u][i];
				for (int j=0; j<J; j++) { // position j
					int w = w_uij[u][i][j];
					int z = (int) (Z * Math.random());
					z_uij[u][i][j] = z;
					N_zu[z][u]++;
					N_zd[z][d]++;
					N_wz[w][z]++;
					N_u[u]++;
					N_d[d]++;
					N_z[z]++;
				}	
			}
		}
		
		// perform Gibbs sampling
		for (int iteration=0; iteration<(burnin+samples); iteration++) {
			for (int u=0; u<U; u++) {  // user u
				for (int i=0; i<w_uij[u].length; i++) { // query i
					int d = d_ui[u][i];
					for (int j=0; j<w_uij[u][i].length; j++) { // position j
						int w = w_uij[u][i][j];
						int z = z_uij[u][i][j];
						
						// remove last value  
						N_zu[z][u]--;
						N_zd[z][d]--;
						N_wz[w][z]--;
						N_z[z]--;
						// calculate distribution
						double[] p = new double[Z];
						double total = 0;
						for (z=0; z<Z; z++) {
							p[z] = ( (N_wz[w][z] + bw_on_W)/(N_z[z] + bw) ) * ( (N_zd[z][d] + bd_on_D)/(N_z[z] + bd) ) * (N_zu[z][u] + a_on_Z);
							total += p[z];
						}
						// resample 
						double val = total * Math.random();
						z = 0; while ((val -= p[z]) > 0) z++;
						// update latent variable and counts
						z_uij[u][i][j] = z;
						N_zu[z][u]++;
						N_zd[z][d]++;
						N_wz[w][z]++;
						N_z[z]++;
					}
				}	
			}	
			
			if (iteration>=burnin) {
				// update parameter estimates
				for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] += (N_wz[w][z] + bw_on_W)/(N_z[z] + bw);
				for (int u=0; u<U; u++)	for (int z=0; z<Z; z++)	P_z_u[z][u] += (N_zu[z][u] + a_on_Z)/(N_u[u] + a);
				for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_d_z[d][z] += (N_zd[z][d] + bd_on_D)/(N_z[z] + bd);
			}
			
			if (iteration%step==0) System.out.println("PTM2: iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}
		
		// normalize parameter estimates
		for (int w=0; w<W; w++) for (int z=0; z<Z; z++) P_w_z[w][z] /= samples;
		for (int u=0; u<U; u++)	for (int z=0; z<Z; z++)	P_z_u[z][u] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_d_z[d][z] /= samples;
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.saveMatrix(P_w_z,"P_w_z.data");
		TopicModelUtils.saveMatrix(P_d_z,"P_d_z.data");
		TopicModelUtils.saveMatrix(P_z_u,"P_z_u.data");
		
		return P_w_z;
	}	
		
	
	
	public double logLikelihood() {
		double ll = 0.0;
		for (int u=0; u<U; u++) {  // user u
			for (int i=0; i<w_uij[u].length; i++) { // query i
				int d = d_ui[u][i];
				for (int j=0; j<w_uij[u][i].length; j++) { // position j
					int z = z_uij[u][i][j];
					int w = w_uij[u][i][j];
					ll += Math.log( ( (N_wz[w][z] + bw_on_W)/(N_z[z] + bw) ) * ( (N_zd[z][d] + bd_on_D)/(N_z[z] + bd) ) * ( (N_zu[z][u] + a_on_Z) / (N_u[u] + a) ) );
				}
			}
		}
		return ll;
	}
	
	
}
