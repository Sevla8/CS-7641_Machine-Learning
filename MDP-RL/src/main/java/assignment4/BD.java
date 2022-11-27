package assignment4;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;

public class BD{

	public static void main(String[] args) throws IOException {
		int[] level = {1, 2, 3};
		double[] reward = {-0.1, 1, 3, 5, 10, 100};
		double[] discount = {0.99, 0.95, 0.9, 0.8, 0.6};
		double[] qInit = {0.3, 0.5, 1, 5, 30};
		double[] epsilon = {0.05, 0.1, 0.3, 0.5, 0.8};
		double[] learningrate = {0.01, 0.1, 0.3, 0.5, 0.9};
		OutputStream outs = System.out;
		PrintStream dos = new PrintStream(outs);

		String head = "level";
		System.setOut(dos);
		System.out.println(head);
  		new File("./stats/"+head).mkdirs();
		File file = new File("./stats/"+head+"/out_BD2"); //Your file
		FileOutputStream fos = new FileOutputStream(file);
		PrintStream ps = new PrintStream(fos);
		System.setOut(ps);
		for(int i=0;i<level.length;i++){
			System.out.println("----------------------"+head+"_"+level[i]+"-----------------------");
			MyBlockDude bd = new MyBlockDude(5,-0.1,0.99,level[i]);
			System.out.println("-------------Value Iteration-------------");
			bd.valueIteration("./stats/"+head+"/BlockDude_"+level[i]);
			System.out.println("-------------Policy Iteration-------------");
			bd.policyIteration("./stats/"+head+"/BlockDude_"+level[i]);
		}

		head = "reward";
		System.setOut(dos);
		System.out.println(head);
		new File("./stats/"+head).mkdirs();
		file = new File("./stats/"+head+"/out_BD"); //Your file
		fos = new FileOutputStream(file);
		ps = new PrintStream(fos);
		System.setOut(ps);
		for(int i=0;i<reward.length;i++){
			System.out.println("----------------------"+head+"_"+reward[i]+"-----------------------");;
			MyBlockDude bd = new MyBlockDude(reward[i],-0.1, 0.99,1);
			System.out.println("-------------Value Iteration-------------");
			bd.valueIteration("./stats/"+head+"/BlockDude_"+reward[i]);
			System.out.println("-------------Policy Iteration-------------");
			bd.policyIteration("./stats/"+head+"/BlockDude_"+reward[i]);
		}

		head = "discount";
		System.setOut(dos);
		System.out.println(head);
		new File("./stats/"+head).mkdirs();
		file = new File("./stats/"+head+"/out_BD"); //Your file
		fos = new FileOutputStream(file);
		ps = new PrintStream(fos);
		System.setOut(ps);
		for(int i=0;i<discount.length;i++){
			System.out.println("----------------------"+head+"_"+discount[i]+"-----------------------");
			MyBlockDude bd = new MyBlockDude(5, -0.1, discount[i],1);
			System.out.println("-------------Value Iteration-------------");
			bd.valueIteration("./stats/"+head+"/BlockDude_"+discount[i]);
			System.out.println("-------------Policy Iteration-------------");
			bd.policyIteration("./stats/"+head+"/BlockDude_"+discount[i]);
		}

		head = "qInit";
		System.setOut(dos);
		System.out.println(head);
		new File("./stats/"+head).mkdirs();
		file = new File("./stats/"+head+"/out_BD"); //Your file
		fos = new FileOutputStream(file);
		ps = new PrintStream(fos);
		System.setOut(ps);
		for(int i=0;i<qInit.length;i++){
			System.out.println("----------------------"+head+"_"+qInit[i]+"-----------------------");
			MyBlockDude bd = new MyBlockDude(5, -0.1, 0.99,1);
			System.out.println("-------------Q Learning-------------");
			bd.QLearning("./stats/"+head+"/BlockDude_"+qInit[i], qInit[i], 1, 0.1);
		}

		head = "epsilon";
		System.setOut(dos);
		System.out.println(head);
		new File("./stats/"+head).mkdirs();
		file = new File("./stats/"+head+"/out_BD"); //Your file
		fos = new FileOutputStream(file);
		ps = new PrintStream(fos);
		System.setOut(ps);
		for(int i=0;i<epsilon.length;i++){
			System.out.println("----------------------"+head+"_"+epsilon[i]+"-----------------------");
			MyBlockDude bd = new MyBlockDude(5, -0.1, 0.99,1);
			System.out.println("-------------Q Learning-------------");
			bd.QLearning("./stats/"+head+"/BlockDude_"+epsilon[i], 0.3, 1, epsilon[i]);
		}

		head = "learningrate";
		System.setOut(dos);
		System.out.println(head);
		new File("./stats/"+head).mkdirs();
		file = new File("./stats/"+head+"/out_BD"); //Your file
		fos = new FileOutputStream(file);
		ps = new PrintStream(fos);
		System.setOut(ps);
		for(int i=0;i<learningrate.length;i++){
			System.out.println("----------------------"+head+"_"+learningrate[i]+"-----------------------");
			MyBlockDude bd = new MyBlockDude(5, -0.1, 0.99,1);
			System.out.println("-------------Q Learning-------------");
			bd.QLearning("./stats/"+head+"/BlockDude_"+learningrate[i], 0.3, learningrate[i], 0.1);
		}

		System.setOut(dos);
		System.out.println("ALL DONE");
	}
}
