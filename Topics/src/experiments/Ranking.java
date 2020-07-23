package experiments;

import java.util.ArrayList;

import data.ModelLoader;

public class Ranking {
	
	
	/*
	 * want to estimate the probability p(d|q)
	 * p(d|q) \propto p(q|d)p(d) = p(d)\prod_{w\in q}p(w|d) = p(d)\prod_{w\in q}\sum_z p(w|z)p(z|d)
	 */ 
	
	public static double[] lda_docScores(int[] query, ModelLoader m) {
		
		int Z = m.P_z_d.length;
		int D = m.P_z_d[0].length;
		
		double[] s = new double[D];
		
		for (int d=0; d<D; d++) {
			// calculate the product over the query terms
			double prod = 1.0;
			for (int w: query) {
				double sum = 0.0;
				for (int z=0; z<Z; z++)
					sum += m.P_w_z[w][z]*m.P_z_d[z][d];

				prod *= sum;
			}			
			
			s[d] = m.P_d[d]*prod;
			
			//if (s[d] == 0.0) System.err.println("Ranking: ERROR score == 0.0");

		}
		
		return s;
	}
	

	
	/*  
	 *  FOR ALL PTM MODELS:
	 *  
	 *  p(d|q,u) \propto p(q,d|u) = p(q|d,u)p(d|u) = p(d|u) \prod_{w\in q} p(w|d,u)
	 *  
	 *  
	 *  FOR THE PTM1 MODEL:
	 *  
	 *  p(d|u) \propto p(u|d)p(d) = p(d) \sum_z p(u|z)p(z|d)
	 *  
	 *  p(w|u,d) = p(w,u|d)/p(u|d)
	 *           = ( \sum_z p(w|z)p(u|z)p(z|d) )/( \sum_z p(u|z)p(z|d) ) 
	 */
	
	public static double[] ptm1_docScores(int[] query, int u, ModelLoader m) {
		
		int D = m.P_d.length;
		int Z = m.P_z_d.length;
		
		double[] s = new double[D];
		
		for (int d=0; d<D; d++) {
			// for each document calculate: SUM_z p(u|z)p(z|d)
			double sum1 = 0;
			for (int z=0; z<Z; z++)
				sum1 += m.P_u_z[u][z]*m.P_z_d[z][d];
			
			// calculate the product over the query terms
			double prod = 1.0;
			for (int w: query) {
				// calculate \sum_z p(w|z)p(u|z)p(z|d)
				double sum2 = 0;
				for (int z=0; z<Z; z++) 
					sum2 += m.P_w_z[w][z]*m.P_u_z[u][z]*m.P_z_d[z][d];
				
				prod *= (sum2/sum1);
			}			
			
			s[d] = m.P_d[d] * sum1 * prod;
			
			//if (s[d] == 0.0) System.err.println("Ranking: ERROR score == 0.0");
		}
		return s;
	}

	
	
	/*
	 *  p(d|u) = \sum_z p(d|z)p(z|u)
	 *  
	 *  p(w|d,u) = p(w,d|u)/p(d|u)
	 *           = ( \sum_z p(w|z)p(d|z)p(z|u) )/( \sum_z p(d|z)p(z|u) ) 
	 */
	
	public static double[] ptm2_docScores(int[] query, int u, ModelLoader m) {
		
		int D = m.P_d_z.length;
		int Z = m.P_d_z[0].length;
		
		double[] s = new double[D];
		
		for (int d=0; d<D; d++) {
			// for each document calculate: SUM_z p(d|z)p(z|u)
			double sum1 = 0;
			for (int z=0; z<Z; z++)
				sum1 += m.P_d_z[d][z]*m.P_z_u[z][u];
			
			// calculate the product over the query terms
			double prod = 1.0;
			for (int w: query) {
				// calculate \sum_z p(w|z)p(d|z)p(z|u)
				double sum2 = 0;
				for (int z=0; z<Z; z++) 
					sum2 += m.P_w_z[w][z]*m.P_d_z[d][z]*m.P_z_u[z][u];
				
				prod *= (sum2/sum1);
			}			
			
			s[d] = sum1 * prod;
			
			//if (s[d] == 0.0) System.err.println("Ranking: ERROR score == 0.0");
		}
		return s;
	}

	
	
	

	

	/*
	 *  p(d|u) = p(d) 
	 *  p(w|d,u) = \sum_{y,z} p(w|y,z)p(z|d)p(y|u) 
	 */
	
	public static double[] ptm3_docScores(int[] query, int u, ModelLoader m) {
		
		int Z = m.P_z_d.length;
		int Y = m.P_y_u.length;
		int D = m.P_z_d[0].length;

		double[] s = new double[D];
		
		for (int d=0; d<D; d++) {
			
			// calculate the product over the query terms
			double prod = 1.0;
			for (int w: query) {
				// of \sum_{y,z} p(w|y,z)p(z|d)p(y|u) 
				double sum2 = 0;
				for (int y=0; y<Y; y++) 
					for (int z=0; z<Z; z++) 
						sum2 += m.P_w_yz[w][y][z]*m.P_z_d[z][d]*m.P_y_u[y][u];
				
				prod *= sum2;
			}			
			
			s[d] = m.P_d[d] * prod;
			
			//if (s[d] == 0.0) System.err.println("Ranking: ERROR score == 0.0");
		}
		
		return s;
	}
	
	
	
	
	/*
	 *  p(d|u) = p(d)        
	 *  p(w|d,u) = \sum_{x,y,z} p(w|x)p(x|y,z)p(z|d)p(y|u) 
	 */
	public static double[] ptm4_docScores(int[] query, int u, ModelLoader m) {
		
		int Z = m.P_z_d.length;
		int Y = m.P_y_u.length;
		int X = m.P_x_yz.length;
		int D = m.P_z_d[0].length;

		double[] s = new double[D];
		
		for (int d=0; d<D; d++) {
			
			// calculate the product over the query terms
			double prod = 1.0;
			for (int w: query) {
				// of \sum_{x,y,z} p(w|x)p(x|y,z)p(z|d)p(y|u) 
				double sum2 = 0;
				for (int x=0; x<X; x++) 
					for (int y=0; y<Y; y++) 
						for (int z=0; z<Z; z++) 
							sum2 += m.P_w_x[w][x]*m.P_x_yz[x][y][z]*m.P_z_d[z][d]*m.P_y_u[y][u];

				prod *= sum2;
				
			}			
			
			s[d] = m.P_d[d] * prod;
			
			//if (s[d] == 0.0) System.err.println("Ranking: ERROR score == 0.0");
		}
		
		return s;
	}
	
	
	// this version maintains a list of top doc scores and only guarantees the top docs have the right score
	public static double[] ptm4_docScores(int[] query, int u, ModelLoader m, boolean sparse, int maxRank) {
		
		if (!sparse) return ptm4_docScores(query, u, m);
		
		ArrayList<Double> topScores = new ArrayList<Double>(maxRank);
		double minScore = 0.0; // lowest score in topScores list
		int minScoreIndex = -1;
		
		int D = m.P_z_d[0].length;

		double[] s_d = new double[D];
		
		for (int d=0; d<D; d++) {
			
			double score = m.P_d[d];
			if (score < minScore) continue; // this doc won't effect the ranking so move to next 
			
			// calculate the product over the query terms
			for (int w: query) {
				// of \sum_{x,y,z} p(w|x)p(x|y,z)p(z|d)p(y|u) 
				double sum2 = 0;
				// calculate sum using sparse data structure
				for (int i=0; i<m.sparseP_x_yz.x.size(); i++) {
					int x = m.sparseP_x_yz.x.get(i);
					int y = m.sparseP_x_yz.y.get(i);
					int z = m.sparseP_x_yz.z.get(i);
					double P_x_yz = m.sparseP_x_yz.val.get(i);
					sum2 += m.P_w_x[w][x]*P_x_yz*m.P_z_d[z][d]*m.P_y_u[y][u];
				}
				score *= sum2;
			
				if (score < minScore) break; // this doc won't effect the ranking so move to next 				
			}			
			
			if (topScores.size() < maxRank) {
				if (score > minScore) {
					minScore = score;
					minScoreIndex = topScores.size();
				}
				topScores.add(score);
			}
			else if (score > minScore) {
				topScores.set(minScoreIndex, score);
				for (int i=0; i<maxRank; i++) if (minScore > topScores.get(i)) {
					minScore = topScores.get(i);
					minScoreIndex = i;
				}
			}
			
			s_d[d] = score;
			
			//if (s_d[d] == 0.0) System.err.println("Ranking: ERROR score == 0.0");
		}
		
		return s_d;
	}
	
	
	
	
	
	/*
	 *  p(d|u) = \sum_z p(d|z)p(z|u)
	 *    	   = \sum_z (p(z|d)p(d)/p(z))p(z|u)
	 *         = p(d) \sum_z p(z|d)p(z|u)/p(z)
	 *         
	 *  p(w|d,u) = \sum_z p(w|z)p(z|d,u) 
	 *   		 
	 *  p(z|d,u) = p(d,u|z)p(z)/p(d,u)
	 *           = p(d|z)p(u|z)p(z)/p(d,u)
	 *           = ( p(z|d)p(d)/p(z) )( p(z|u)p(u)/p(z) ) p(z)/p(d,u)
	 *           = ( p(z|d)p(z|u)/p(z) )( p(d)p(u)/p(d,u) )
	 *           = ( p(z|d)p(z|u)/p(z) ) / ( \sum_z p(z|d)p(z|u)/p(z) )
	 *  
	 *  Hence:
	 *  p(w|d,u) = ( \sum_z p(w|z)p(z|d)p(z|u)/p(z) ) / ( \sum_z p(z|d)p(z|u)/p(z) )
	 */
	
	public static double[] ttm1_docScores(int[] query, int u, ModelLoader m) {
		
		int Z = m.P_z_d.length;
		int D = m.P_z_d[0].length;
		
		double[] s = new double[D];
		
		for (int d=0; d<D; d++) {
			// for each document calculate: SUM_z p(z|d)p(z|u)/p(z)
			double sum1 = 0;
			for (int z=0; z<Z; z++)
				sum1 += m.P_z_d[z][d]*m.P_z_u[z][u]/m.P_z[z];
			
			// calculate the product over the query terms
			double prod = 1.0;
			for (int w: query) {
				// calculate \sum_z p(w|z)p(z|d)p(z|u)/p(z)
				double sum2 = 0;
				for (int z=0; z<Z; z++)
					sum2 += m.P_w_z[w][z]*m.P_z_d[z][d]*m.P_z_u[z][u]/m.P_z[z];
				
				prod *= (sum2/sum1);
			}			
			
			s[d] = m.P_d[d] * sum1 * prod;
		}
		
		return s;
	}

	
	
	
}
