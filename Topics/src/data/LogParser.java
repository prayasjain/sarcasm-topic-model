package data;

import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

import utils.QueryTokenizer;


public class LogParser {
	
	
	public static QueryLog parseAOLLog(String inputTable, int maxQueriesPerUser, int minQueriesPerTerm) {
		
		// parameter settings
		boolean removeStopwords = false;
		boolean doStemming = true;
		boolean removePunctuation = true;
		
		
		// write out parameter settings to a log file:		
		try {
			PrintWriter p = new PrintWriter(new FileOutputStream("parseinfo.txt"));
			p.println("inputTable = "+inputTable);
			p.println("minQueriesPerTerm = "+minQueriesPerTerm);
			p.println("removeStopwords = "+removeStopwords);
			p.println("doStemming = "+doStemming);
			p.println("removePunctuation = "+removePunctuation);
			p.close();
		}
		catch (Exception e) {e.printStackTrace(); System.exit(9);}
		
		// JDBC connection parameters and driver
		String jdbcURL = "jdbc:mysql://localhost/mark?user=mark&password=taipei101";
		String jdbcDriver = "com.mysql.jdbc.Driver";
		java.sql.Connection con;
		
		
		// load tokenizer
		QueryTokenizer qt = new QueryTokenizer(removeStopwords, doStemming, removePunctuation);
				
		try {
				
			// connect to the database
			Class.forName(jdbcDriver).newInstance();
			con = java.sql.DriverManager.getConnection(jdbcURL);
			
			// FIRST PASS: calculate term frequencies
			HashMap<String,Integer> queryFreqForTerm = new HashMap<String,Integer>();
			
			String q1 = "SELECT Query FROM "+inputTable+";";
			PreparedStatement ps1 = con.prepareStatement(q1);
			ResultSet rs1 = ps1.executeQuery();
			while (rs1.next()) { 
				String query = rs1.getString(1);
				ArrayList<String> previousTermsInCurrentQuery = new ArrayList<String>();
				for (String t: qt.tokenize(query)) {
					if (previousTermsInCurrentQuery.contains(t)) continue;
					else previousTermsInCurrentQuery.add(t);
					Integer count = queryFreqForTerm.get(t);
					if (count == null) count = 0;
					queryFreqForTerm.put(t,count+1);
				}	
			}
			ps1.close();
			
			// put lexicon in array
			class StringInt implements Comparable<StringInt> {
				String s;
				int i;
				public StringInt(String s, int i) {this.s = s; this.i = i;}
				public int compareTo(StringInt si) { return - (new Integer(i)).compareTo(si.i); }
			}
			ArrayList<StringInt> lexicon = new ArrayList<StringInt>();
			for (String t: queryFreqForTerm.keySet()) {
				int count = queryFreqForTerm.get(t);
				if (count >= minQueriesPerTerm) lexicon.add(new StringInt(t,count));
			}
			// sort the lexicon
			Collections.sort(lexicon);
			// create reverse mapping from terms to termIDs
			HashMap<String,Integer> termIDs = new HashMap<String,Integer>();
			int id = 0; 
			for (StringInt sip: lexicon) termIDs.put(sip.s, id++);
			
			// SECOND PASS: build word and doc matrices (using ArrayLists)
			
			String q2 = "SELECT MAX(UserID), MAX(UrlID) FROM "+inputTable+";";
			PreparedStatement ps2 = con.prepareStatement(q2);
			ResultSet rs2 = ps2.executeQuery();
			rs2.next();
			int userCount = rs2.getInt(1); // note: count == max since ids start from 1 not 0 
			int docCount = rs2.getInt(2);
			ps2.close();
			
			ArrayList<ArrayList<ArrayList<Integer>>> words_uij = new ArrayList<ArrayList<ArrayList<Integer>>>(userCount);
			ArrayList<ArrayList<Integer>> docs_ui = new ArrayList<ArrayList<Integer>>(userCount);
			ArrayList<ArrayList<Long>> time_ui = new ArrayList<ArrayList<Long>>(userCount);
			
			// initialize arraylists for all users:
			for (int u=0; u<userCount; u++) {
				words_uij.add(new ArrayList<ArrayList<Integer>>());
				docs_ui.add(new ArrayList<Integer>());
				time_ui.add(new ArrayList<Long>());
			}
			
			String q3 = "SELECT Query, UserID, UrlID, QueryTime FROM "+inputTable+" ORDER BY QueryTime;";
			PreparedStatement ps3 = con.prepareStatement(q3);
			ResultSet rs3 = ps3.executeQuery();
			while (rs3.next()) { 
								
				String query  = rs3.getString(1);
				int userIndex = rs3.getInt(2) - 1; // note that: index == id-1 since ids start from 1 not 0
				int urlIndex  = rs3.getInt(3) - 1;
				long timestamp = rs3.getTimestamp(4).getTime();
				
				// check if we already have enough (max) queries for that user
				if (words_uij.get(userIndex).size()>=maxQueriesPerUser) continue;
				
				ArrayList<Integer> queryWords = new ArrayList<Integer>();
				for (String t: qt.tokenize(query)) {
					Integer termID = termIDs.get(t);
					if (termID != null) queryWords.add(termID); 
				}
				
				if (queryWords.size() > 0) {
					words_uij.get(userIndex).add(queryWords);
					docs_ui.get(userIndex).add(urlIndex);
					time_ui.get(userIndex).add(timestamp);
				}
				
			}
			ps1.close();

			// generate return object
			QueryLog l = new QueryLog();
			l.U = userCount;
			l.W = termIDs.size();
			l.D = docCount;
			
			// convert ArrayLists to Arrays:
			l.w_uij = new int[userCount][][];
			l.d_ui  = new int[userCount][];
			l.t_ui  = new long[userCount][];
			l.N = 0;
			for (int u=0; u<userCount; u++) {
				ArrayList<ArrayList<Integer>> words_ij = words_uij.get(u);
				ArrayList<Integer> docs_i = docs_ui.get(u);
				ArrayList<Long> time_i = time_ui.get(u);
				int queryCount = docs_i.size();
				l.w_uij[u] = new int[queryCount][];
				l.d_ui[u] = new int[queryCount];
				l.t_ui[u] = new long[queryCount];
				for (int i=0; i<queryCount; i++) {
					ArrayList<Integer> query = words_ij.get(i);
					l.w_uij[u][i] = new int[query.size()];
					for (int j=0; j<query.size(); j++) l.w_uij[u][i][j] = query.get(j);
					l.d_ui[u][i] = docs_i.get(i);
					l.t_ui[u][i] = time_i.get(i);
					l.N += query.size();
				}	
			}
			l.l_w = new String[l.W];
			l.N_w = new int[l.W];
			for (int w=0; w<l.W; w++) {
				l.l_w[w] = lexicon.get(w).s;
				l.N_w[w] = lexicon.get(w).i;
			}
			
			return l;
			
		} 
		catch (Exception e) {e.printStackTrace();}
		return null;
	}

	
	
	public static void main (String[] args) {
		
		int minUsersPerURL = Integer.parseInt(args[0]);
		int minQueriesPerUser = Integer.parseInt(args[1].replace("mx",""));
		int maxQueriesPerUser = Integer.MAX_VALUE; if (args[1].contains("mx")) maxQueriesPerUser = minQueriesPerUser;
		int minQueriesPerTerm = Integer.parseInt(args[2]);
		
		// To generate the input table use the SQL script present on Bowmore 
		String inputTable = "ptm_queries_"+minUsersPerURL+"_"+minQueriesPerUser;
		
		System.out.println("LogParser: parsing AOL log ...");
		QueryLog l = LogParser.parseAOLLog(inputTable,maxQueriesPerUser,minQueriesPerTerm);
		l.save("aol");
		System.out.println("LogParser: finished parsing the log.");			
		
	}
	
}
