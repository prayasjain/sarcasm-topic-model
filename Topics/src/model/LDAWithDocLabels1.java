package model;

import java.util.ArrayList;
import java.util.HashMap;

import utils.TopicModelUtils;



public class LDAWithDocLabels1 {
	//Aadi: Assignments of words to topics are the only latent vars here
	public int Z; // number of topics
	public int W; // vocabulary
	public int D; // number of documents
	public long N; // total word occurrences
	public int S; // number of sentiment labels
	
	public double a; // topic smoothing hyperparameter \alpha
	public double b; // term smoothing hyperparameter \beta 
	
	
	public int[][] w_di; // w_di[d][i] = i'th word in the d'th document
	public int[][] z_di; // z_di[d][i] = topic assignment to i'th position in d'th document

	public int[][] N_zd; // N_zd[z][d] = count of z'th topic in d'th document
	public int[][] N_wz; // N_wz[w][z] = count of w'th word for z'th topic
	public int[][] N_zc; //N_zc[z][c] = count of z'th topic in c'th collection
	
	public int[] N_z;    // N_z[z] = count of z'th topic
	public int[] N_d;    // N_d[d] = length of document d
	public int[] N_c;	// N_c[] = number of word occurrences in a collection
	
	public double[][] P_w_z;
	public double[][] P_z_d;
	public double[][] P_z_c;
	
	public int num_samples;
	
	// private variables
	private double b_on_W;
	private double a_on_Z;
	private double ac_on_Z;

	/*
	 * S: Number of labels
	 * hm_senti: hashmap containing word and tag. This is loaded from the sentiment word list
	 * s_i: An array of arraylists, each of which contain doc ids with a given output label.
	 */
	public double[][] estimate(int[][] w_di, int W, int D, long N, int Z, HashMap hm_senti, String[] l_w, ArrayList[] s_i, int S, double a, double b, double ac, int burnIn, int samples, int step) {

		this.w_di = w_di;
		this.Z = Z;
		this.W = W;
		this.D = D;
		this.N = N;
		this.a = a;
		this.b = b;
		this.S = S;
		num_samples = 0;
		b_on_W = b/W;
		a_on_Z = a/Z;
		ac_on_Z = ac/Z;


		// initialize latent variable assignment and count matrices
		z_di = new int[D][];
		N_zd = new int[Z][D];
		N_wz = new int[W][Z];
		N_z = new int[Z];
		N_d = new int[D];
		N_c = new int[S+1];   // The +1 indicates an additional collection if a document does not have a sentiment label
		N_zc = new int[Z][S+1];
		
		P_w_z = new double[W][Z];
		P_z_d = new double[Z][D];
		P_z_c = new double[Z][S+1];
		
		
		for (int d=0; d<D; d++) {  // document d
			
			int I = w_di[d].length;
			int doc_label = 0;

			/*
			 * Get the document label for this document.
			 */
			for (int i = 1; i <= S; i++)
				if (s_i[i].contains(d))
				{
					doc_label = i;
					break;
				}
			
			N_c[doc_label] += I;
			
			z_di[d] = new int[I];
			N_d[d] = I;
			for (int i=0; i<I; i++) { // position i
				int z = (int) (Z * Math.random());
				z_di[d][i] = z;				// Aadi: Initialize each word with a randomly gen. topic
				N_zd[z][d]++;				//Aadi: Increment the corresponding counters. N_zd, N_wz, N_z
				N_wz[w_di[d][i]][z]++;
				N_zc[z][doc_label]++;
				N_z[z]++;
			}
		}

		// perform Gibbs sampling
		for (int iteration=0; iteration<burnIn+samples; iteration++) {    // Aadi: For burn-in + samples number of iterations
			for (int d=0; d<D; d++) { // document d

				int doc_label = -1;

				for (int i = 1; i <= S; i++)
					if (s_i[i].contains(d))
					{
						doc_label = i;
						break;
					}

				if (doc_label == -1)
				{
					System.err.println("The file "+ d +" does not have a sentiment label.");
					continue;
				}

				for (int i=0; i<w_di[d].length; i++) { // position i
					// Aadi: Go over each word of all documents
					int w = w_di[d][i];				// Aadi: Which word is this?
					int z = z_di[d][i];				// Aadi: Which topic is it assigned to??


					// remove last value  			
					N_zd[z][d]--;					// Aadi: Reduce counts corresponding to this word and this topic
					N_wz[w][z]--;
					N_z[z]--;
					N_zc[z][doc_label]--;
					
					// calculate distribution p(z|w,d) /propto p(w|z)p(z|d)
					double[] p = new double[Z];
					double total = 0;
					for (z=0; z<Z; z++) {
						p[z] = ( (N_wz[w][z] + b_on_W)/(N_z[z] + b) ) * (N_zd[z][d] + a*( (N_zc[z][doc_label] + ac*(1/Z))/ N_c[doc_label] + ac ) );
						total += p[z];;
					}

					// resample 
					double val = total * Math.random();	// I dont know how this incorporates the probability?!
					z = 0; while ((val -= p[z]) > 0) z++;  // select a new topic

					// update latent variable and counts
					z_di[d][i] = z;

					N_zd[z][d]++;   // update vars
					N_wz[w][z]++;
					N_z[z]++;
					N_zc[z][doc_label]++;

				}	
			}	

			// update parameter estimates
			if (iteration >= burnIn) {	//Aadi: A sample is a complete configuration of probabilities at the end of an iter.
				for (int w=0; w<W; w++) for (int z=0; z<Z; z++) for (int l=0;l<S+1;l++) P_w_z[w][z] += (N_wz[w][z] + b_on_W)/(N_z[z] + b);
				for (int d=0; d<D; d++) for (int z=0; z<Z; z++) P_z_d[z][d] += (N_zd[z][d] + a_on_Z)/(N_d[d] + a);
				for (int c=0; c<S+1; c++) for (int z=0; z<Z; z++) P_z_c[z][c] += (N_zc[z][c] + ac_on_Z)/(N_c[c] + ac);
			}

			if (iteration%step==0) System.out.println("iteration: "+iteration+", log-likelihood: "+logLikelihood());
		}

		// normalize parameter estimates
		for (int w=0; w<W; w++) for (int z=0; z<Z; z++)for (int l=0;l<S+1;l++)  P_w_z[w][z] /= samples;
		for (int d=0; d<D; d++) for (int z=0; z<Z; z++)	P_z_d[z][d] /= samples;
		for (int c=0; c<S+1; c++) for (int z=0; z<Z; z++) P_z_c[z][c] /= samples;
		
		System.out.println("Saving parameters of model:");
		TopicModelUtils.saveMatrix(P_w_z,"P_w_z.data");
		TopicModelUtils.saveMatrix(P_z_d,"P_z_d.data");
		TopicModelUtils.saveMatrix(P_z_c,"P_z_c.data");
		TopicModelUtils.saveVector(estimateP_d(),"P_d.data");

		return P_w_z;		
	}	


	public double logLikelihood() {
		double ll = 0;
		for (int d=0; d<D; d++) { // document d
			for (int i=0; i<N_d[d]; i++) { // position i
				for (int l = 0; l<S+1;l++){
					int z = z_di[d][i];
					int w = w_di[d][i];
					ll += Math.log( (N_wz[w][z] + b_on_W)/(N_z[z] + b) ); 
					ll += Math.log( (N_zd[z][d] + a_on_Z)/(N_d[d] + a) );
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
