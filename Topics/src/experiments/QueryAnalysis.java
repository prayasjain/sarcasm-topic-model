package experiments;

import java.util.HashSet;

import utils.TopicModelUtils;
import data.QueryLog;
import data.ModelLoader;

public class QueryAnalysis {

	static boolean useSparseDataStructure = true;
	
	static int maxRank; // maximum rank of document
	static int maxUsers; // maximum number of users to evaluate
	static double weight; // weight of user profile in range [0,1]
	
	static double[][] ldaRanks;
	
	
	public static void main (String[] args) {
		
		String model    = args[0];
		String logFile  = args[1];
		String rankFile = args[2];
		maxRank 	    = Integer.parseInt(args[3]);
		maxUsers 	    = Integer.parseInt(args[4]);
		weight			= Double.parseDouble(args[5]);
						
		QueryLog test = new QueryLog();
		test.load(logFile);
		
		// load lexicon so we can print out queries
		test.loadLexicon();
		
		QueryLog train = new QueryLog();
		train.load(logFile.replace("test","train"));
		
		eval(test,model,rankFile,train);
		
	}
	
	
				
	// Run evaluation 
	public static void eval(QueryLog test, String model, String rankFile, QueryLog train) {
				
		int U = (maxUsers>test.U) ? test.U: maxUsers;
		
		double logN = Math.log(train.N+test.N);
		
		ModelLoader m = new ModelLoader(model, useSparseDataStructure, weight);
		
		boolean lda = model.equals("LDA");
		
		if (lda) ldaRanks= new double[U][];
		else ldaRanks = TopicModelUtils.loadNonRectMatrix(rankFile);
		
		System.out.println("success persRank baseRank trainQueryCount trainDistinctUrlCount entropyP_z_u termsInQuery queryLogLikelihood userId queryIndex [query-terms]");
		
		for(int u=0; u<U; u++) {// For each user
			
			int Q = test.w_uij[u].length;
			if (lda) ldaRanks[u] = new double[Q];
			
			// user specific quantities
			int trainQueryCount = train.w_uij[u].length;
			int trainDistinctUrlCount = countDistinct(train.d_ui[u]);
			double entropyP_z_u = 0; 
			if (model.startsWith("PTM2")) entropyP_z_u = entropy(m.P_z_u, u);
			if (model.startsWith("PTM3") || model.startsWith("PTM4")) entropyP_z_u = entropy(m.P_y_u, u);
			
			// query (and user) specific quantities
			double [] score_d_qu = null;
			int termsInQuery = 0;
			double queryLogLikelihood = 0.0;
			String queryAsString = null;
			
			for(int q=0; q<Q; q++) if (test.w_uij[u][q].length > 0) {// For each query
				
				if (q==0 || !sameQuery(test.w_uij[u][q-1],test.w_uij[u][q])) { 
					// (re)calculate document ranking only if query has changed
					if (lda)  score_d_qu = Ranking.lda_docScores(test.w_uij[u][q], m);
					if (model.startsWith("PTM1")) score_d_qu = Ranking.ptm1_docScores(test.w_uij[u][q], u, m);
					if (model.startsWith("PTM2")) score_d_qu = Ranking.ptm2_docScores(test.w_uij[u][q], u, m);
					if (model.startsWith("PTM3")) score_d_qu = Ranking.ptm3_docScores(test.w_uij[u][q], u, m);
					if (model.startsWith("PTM4")) score_d_qu = Ranking.ptm4_docScores(test.w_uij[u][q], u, m, useSparseDataStructure, 25);	
					// same for query-specific information
					termsInQuery = test.w_uij[u][q].length;
					queryLogLikelihood = queryLogLikelihood(test,u,q,logN);
					queryAsString = queryAsString(test,u,q);
				}
								
				int rank = find(utils.Sorter.rankedList(score_d_qu), test.d_ui[u][q]);
				if (lda) ldaRanks[u][q] = rank;
				
				int ldaRank = (int) ldaRanks[u][q];
				int success = successfulResult(rank,ldaRank);
								
				System.out.printf("%3d %3d %3d %6d %6d %g %6d %g %6d %3d %s\n", success, rank, ldaRank, trainQueryCount, trainDistinctUrlCount, entropyP_z_u, termsInQuery, queryLogLikelihood, u, q, queryAsString);
				
			}
		}
		
		if (lda) TopicModelUtils.saveNonRectMatrix(ldaRanks,rankFile); // save LDA ranks

	}
	
	
	static boolean sameQuery(int[] q1, int[] q2) {
		if (q1.length != q2.length) return false;
		for (int j=0; j<q1.length; j++) if (q1[j] != q2[j]) return false;
		return true;
	}
	
	
	static int find(int[] rankedDocs, int d) {
		for(int r=0; r<maxRank; r++)
			if(rankedDocs[r] == d) 
				return r;
		return -1;
	}
	
	
	static int successfulResult(int persRank, int baseRank) {
		// returns 1 if persRank is better than baseRank, 0 if the same and -1 otherwise
		if (persRank == baseRank) return 0;
		if (baseRank == -1) return 1;
		if (persRank == -1) return -1;
		if (persRank < baseRank) return 1;
		else return -1;
	}
	
	
	static int countDistinct(int[] a) {
		HashSet<Integer> s = new HashSet<Integer>();
		for (int i : a) s.add(i);
		return s.size();
	}
	
	
	static double entropy(double[][] P_a_b, int b) {
		double e = 0.0;
		int A = P_a_b.length;
		for (int a=0; a<A; a++)
			e += P_a_b[a][b]*Math.log(P_a_b[a][b]);
		return -e;
	}
	
	
	static double queryLogLikelihood(QueryLog data, int u, int q, double logN) {
		double ll = 0.0;
		for (int w: data.w_uij[u][q]) {
			ll += Math.log(data.N_w[w]) - logN;
		}
		return ll;
	}
	
	
	static String queryAsString(QueryLog data, int u, int q) {
		// output query:
		StringBuffer s = new StringBuffer();
		boolean first = true;
		s.append("[");
		for (int w: data.w_uij[u][q]) {
			if (first) first = false; 
			else s.append(",");
			s.append(data.l_w[w]);
		}
		s.append("]");
		return s.toString();
	}
		
}
