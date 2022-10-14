#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "../inc/hmm.h"

using namespace std;

int main()
{
	//建立一個map對應英文到數字，之後做index使用 
	map<char, int> dict;
	dict['A']=0; dict['B']=1; dict['C']=2; dict['D']=3; dict['E']=4; dict['F']=5;

	HMM hmms[5];
	load_models( "../modellist.txt", hmms, 5);
	dump_models( hmms, 5);
	
	int T=50;
	int N=6;
	float correct_num=0;
	
	ifstream infile("../data/test_seq.txt");
	ifstream label_file("../data/test_lbl.txt");
	string test_line;
	string label_line;
	ofstream outfile ("result.txt");
	while(getline(infile, test_line)){
		double total_max[5]={0};		//用來存五個model最大路徑機率值比大小，判斷此sequence屬於哪個model
		double model_max=0;			//用來存五個model中最大的值 
		string pred_model = "";
		
		for(int q=0; q<5; q++){
			//開始計算到各點的機率矩陣Delta
			double delta[T][N];	
			int idx = 0;
			//初始值
			for(int i=0; i<N; i++){
				idx = dict[test_line[0]];
				delta[0][i] = hmms[q].initial[0] * hmms[q].observation[i][idx];
			}	
			// DP delta
			for(int i=1; i<T; i++){			//對於某觀測值 
				for(int j=0; j<N;j++){		//在某狀態 
					//從上一個col求最大機率值
					double max = 0;
					for(int k=0; k<N; k++){
						if(delta[i-1][k]*hmms[q].transition[k][j] > max)
							max = delta[i-1][k]*hmms[q].transition[k][j];
					}
					
					idx = dict[test_line[i]];
					delta[i][j] = max * hmms[q].observation[j][idx];
				}
			}
			
			//然後求出最後一個col的最大值，之後5個model比大小
			for(int i=0; i<N; i++){
				if(delta[T-1][i] > total_max[q]) total_max[q] = delta[T-1][i];
			}
			
			//然後比較五個model哪個機率最大 
			if(total_max[q] > model_max){
				model_max = total_max[q];
				
				int w = q+1;
				char int2char = w + '0';
				string read_file_name = "model_0";
				read_file_name += int2char;
				read_file_name += ".txt";
				pred_model = read_file_name;
			}
		}
		
		getline(label_file, label_line);
		if(pred_model.compare(label_line) == 0) correct_num++;
//		outfile << pred_model << " " << model_max << endl;
	}
	cout << "accurracy = " << correct_num/2500 << endl;
	
	infile.close();
	outfile.close();
	label_file.close();
	
	return 0;
}
