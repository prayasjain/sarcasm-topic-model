package data;

import utils.Sparse3dMatrix;
import utils.TopicModelUtils;

public class ModelLoader 
{
	// LDA
	public double[][] P_w_z;
	public double[][] P_z_d;
	public double[] P_d;
	
	// PTM1
	public double[][] P_u_z;
	
	// PTM2
	public double[][] P_z_u;
	public double[][] P_d_z;
	
	// PTM3
	public double[][][] P_w_yz;
	public double[][] P_y_u;
	
	// PTM4
	public double[][] P_w_x;
	public double[][][] P_x_yz;
	public Sparse3dMatrix sparseP_x_yz;
	
	// TTM1
	public double[] P_z;
	
	
	boolean sparse = false;
	double sparsityCutoff = 1.0;
	double weight = 1.0;
	
	
	public ModelLoader(String model, boolean useSparseDataStructure, double profileWeight)
	{
		weight = profileWeight;
		sparse = useSparseDataStructure;

		if (model.equals("LDA"))  loadLDA();// Load saved parameters for LDA model
		if (model.startsWith("PTM1")) loadPTM1();// Load saved parameters for PTM1 model
		if (model.startsWith("PTM2")) loadPTM2();// Load saved parameters for PTM2 model
		if (model.startsWith("PTM3")) loadPTM3();// Load saved parameters for PTM3 model
		if (model.startsWith("PTM4")) loadPTM4();// Load saved parameters for PTM4 model
		if (model.startsWith("TTM1")) loadTTM1();// Load saved parameters for TTM1 model
	}
	
	
	public void loadLDA()
	{
		P_w_z = TopicModelUtils.loadMatrix("P_w_z.data");
		P_z_d = TopicModelUtils.loadMatrix("P_z_d.data");
		P_d = TopicModelUtils.loadVector("P_d.data");
	}
	
	
	public void loadPTM1()
	{
		P_w_z = TopicModelUtils.loadMatrix("P_w_z.data");
		P_u_z = TopicModelUtils.loadMatrix("P_u_z.data");
		P_z_d = TopicModelUtils.loadMatrix("P_z_d.data");
		P_d = TopicModelUtils.loadVector("P_d.data");
	
		if (weight != 1.0) 
			for (int u=0; u<P_u_z.length; u++)
				for (int z=0; z<P_u_z[u].length; z++)
					P_u_z[u][z] = Math.pow(P_u_z[u][z], weight);
	}
	
	
	public void loadPTM2()
	{
		P_w_z = TopicModelUtils.loadMatrix("P_w_z.data");
		P_d_z = TopicModelUtils.loadMatrix("P_d_z.data");
		P_z_u = TopicModelUtils.loadMatrix("P_z_u.data");

		if (weight != 1.0) 
			for (int z=0; z<P_z_u.length; z++)
				for (int u=0; u<P_z_u[z].length; u++)
					P_z_u[z][u] = Math.pow(P_z_u[z][u], weight);
	}
	
	
	public void loadPTM3()
	{
		P_w_yz = TopicModelUtils.load3dMatrix("P_w_yz.data");
		P_z_d = TopicModelUtils.loadMatrix("P_z_d.data");
		P_y_u = TopicModelUtils.loadMatrix("P_y_u.data");
		P_d = TopicModelUtils.loadVector("P_d.data");
		
		if (weight != 1.0) 
			for (int y=0; y<P_y_u.length; y++)
				for (int u=0; u<P_y_u[y].length; u++)
					P_y_u[y][u] = Math.pow(P_y_u[y][u], weight);
	}
	
	
	public void loadPTM4()
	{
		if (sparse) sparseP_x_yz = TopicModelUtils.loadSparse3dMatrix("P_x_yz.data", sparsityCutoff);
		else P_x_yz = TopicModelUtils.load3dMatrix("P_x_yz.data");
		
		P_w_x = TopicModelUtils.loadMatrix("P_w_x.data");
		P_z_d = TopicModelUtils.loadMatrix("P_z_d.data");
		P_y_u = TopicModelUtils.loadMatrix("P_y_u.data");
		P_d = TopicModelUtils.loadVector("P_d.data");

		if (weight != 1.0) 
			for (int y=0; y<P_y_u.length; y++)
				for (int u=0; u<P_y_u[y].length; u++)
					P_y_u[y][u] = Math.pow(P_y_u[y][u], weight);
	}
	
	
	public void loadTTM1()
	{
		P_w_z = TopicModelUtils.loadMatrix("P_w_z.data");
		P_z_d = TopicModelUtils.loadMatrix("P_z_d.data");
		P_z_u = TopicModelUtils.loadMatrix("P_z_u.data");
		P_d = TopicModelUtils.loadVector("P_z.data");
		P_d = TopicModelUtils.loadVector("P_d.data");
	}
	
		
}
