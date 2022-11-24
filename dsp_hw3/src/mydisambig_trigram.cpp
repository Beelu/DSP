#include <stdio.h>
#include "Ngram.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include<math.h>
using namespace std;

Vocab voc;
Ngram lm( voc, 3 );

// 動態分配3d矩陣
int ***create_3D_Array(int a, int b, int c){
	int ***A = new int**[a];
	for (int i = 0 ; i < a ; i++){
		A[i] = new int*[b];
		for (int j = 0 ; j < b ; j++)
		{
			A[i][j] = new int[c];
		}
	}
	for (int i = 0 ; i < a ; i++)
		for (int j = 0 ; j < b ; j++)
			for (int k = 0 ; k < c ; k++)
				A[i][j][k] = 0;
	return A;
}
double ***create_3D_Array_2(int a, int b, int c){
	double ***A = new double**[a];
	for (int i = 0 ; i < a ; i++){
		A[i] = new double*[b];
		for (int j = 0 ; j < b ; j++)
		{
			A[i][j] = new double[c];
		}
	}
	for (int i = 0 ; i < a ; i++)
		for (int j = 0 ; j < b ; j++)
			for (int k = 0 ; k < c ; k++)
				A[i][j][k] = 0.0;
	return A;
}

void delete_3D_Array(int ***arr, int a, int b, int c){
	for (int i = 0 ; i < a ; i++){
		for (int j = 0 ; j < b ; j++){
			delete [] arr[i][j];
		}
		delete[] arr[i];
	}
	delete [] arr;
}
void delete_3D_Array_2(double ***arr, int a, int b, int c){
	for (int i = 0 ; i < a ; i++){
		for (int j = 0 ; j < b ; j++){
			delete [] arr[i][j];
		}
		delete[] arr[i];
	}
	delete [] arr;
}

// Get P(W2 | W1) -- bigram
double getBigramProb(const char *w1, const char *w2)
{
    VocabIndex wid1 = voc.getIndex(w1);
    VocabIndex wid2 = voc.getIndex(w2);

    if(wid1 == Vocab_None)  //OOV
        wid1 = voc.getIndex(Vocab_Unknown);
    if(wid2 == Vocab_None)  //OOV
        wid2 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = { wid1, Vocab_None };
    return lm.wordProb( wid2, context);
}

// Get P(W3 | W1, W2) -- trigram
double getTrigramProb(const char *w1, const char *w2, const char *w3) 
{
    VocabIndex wid1 = voc.getIndex(w1);
    VocabIndex wid2 = voc.getIndex(w2);
    VocabIndex wid3 = voc.getIndex(w3);

    if(wid1 == Vocab_None)  //OOV
        wid1 = voc.getIndex(Vocab_Unknown);
    if(wid2 == Vocab_None)  //OOV
        wid2 = voc.getIndex(Vocab_Unknown);
    if(wid3 == Vocab_None)  //OOV
        wid3 = voc.getIndex(Vocab_Unknown);

    VocabIndex context[] = { wid2, wid1, Vocab_None };
    return lm.wordProb( wid3, context );
}

//============================以下主程式=======================
int main(int argc, char *argv[]){
    if(argc<5){
        cout << "input data not enough: "<<argc << endl;
		return 0;
	}
    
    {
        const char* lm_filename[] = {argv[3]};
        File lmFile( *lm_filename, "r" );
        lm.read(lmFile);
        lmFile.close();
    }

    // 讀檔
    ifstream ex_file(argv[1]);
    ifstream map_file(argv[2]);
    ofstream outfile(argv[4]);

    // 將map中的key跟value存在vector中用來做之後的對應
    // map_key = 每個可能出現的國字及注音 (ex:ㄅ ㄆ ㄇ...)
    // map_values = map_key對應到的所有可能國字(ex:[八, 匕, 卜...], [仆, 匹, 片...])
    string map_line = "";
    vector<string> map_key;
    vector< vector<string> > map_values;
    while(getline(map_file, map_line)){
        map_key.push_back(map_line.substr(0,2));

        vector<string> map_value;
        for (int i = 2; i < map_line.length(); i++) {
            if (map_line[i] == ' ' || map_line[i] == '\t') continue;
            map_value.push_back(map_line.substr(i++,2));
        }
        // map_value.pop_back();
        map_values.push_back(map_value);
    }
    //=================================
    string ex_line = "";
    while(getline(ex_file, ex_line)){
        // example每行字串轉為矩陣
        vector<string> word_vec;
        for (int i = 0; i < ex_line.length(); i++) { 
            if (ex_line[i] == ' ') continue;
            word_vec.push_back(ex_line.substr(i++,2));
        }

        // 找在example中每個字可以map到的中文字
        // all_word_maps = 該字串中每個字對應到的map_value (ex:若第二個字是ㄅ 則all_word_maps[1]=[八, 匕, 卜...])
        vector< vector<string> > all_word_maps;
        for(int i=0; i<word_vec.size(); i++){
            for(int j=0; j<map_key.size(); j++){
                if(map_key[j].compare(word_vec[i])==0){            // 找到該字對應的map_key
                    all_word_maps.push_back(map_values[j]);
                    break;
                }
            }
        }
        vector<string> end;
        end.push_back("</s>");
        all_word_maps.push_back(end);
        
        // Viterbi
        // 建立矩陣
        // trigram中用三圍矩陣，prob[k][i][j]=在input第k個字是i且第k-1個字是j的機率
        // last_pos[k][i][j]=在input第k個字是i且第k-1個字是j的最大路徑下，紀錄上一個也就是第k-2個字
        int row = all_word_maps.size();
        int col = 0;                            //這裡我們取對應字最多的注音當作col長度
        for(int i=0; i<all_word_maps.size(); i++){
            if(all_word_maps[i].size()>col)
                col = all_word_maps[i].size();
        }

        int ***last_pos = create_3D_Array(row, col, col);
        double ***prob = create_3D_Array_2(row, col, col);
        
        //=======正式開始算==========
        //初始值(backtrack時理論上不會用到這兩個的last_pos，因此不初始化)
        for(int i=0; i<all_word_maps[0].size(); i++){
            for(int j=0; j<all_word_maps[0].size(); j++){
                prob[0][i][j] = getBigramProb("<s>", all_word_maps[0][i].c_str());
            }
        }
        for(int i=0; i<all_word_maps[1].size(); i++){
            for(int j=0; j<all_word_maps[0].size(); j++){
                prob[1][i][j] = prob[0][j][0] + getTrigramProb("<s>", all_word_maps[0][j].c_str(), all_word_maps[1][i].c_str());
            }
        }

        for(int k=2; k<row; k++){                                       // 走訪一行中的每個字
            for(int i=0; i<all_word_maps[k].size(); i++){             // 走訪這個字的所有可能
                for(int j=0; j<all_word_maps[k-1].size(); j++){           // 走訪上個字的所有可能
                    prob[k][i][j] = prob[k-1][j][0] + getTrigramProb(all_word_maps[k-2][0].c_str(), all_word_maps[k-1][j].c_str(), all_word_maps[k][i].c_str());
                    last_pos[k][i][j] = 0;
                    for(int q=1; q<all_word_maps[k-2].size(); q++){
                        double temp_p = prob[k-1][j][q] + getTrigramProb(all_word_maps[k-2][q].c_str(), all_word_maps[k-1][j].c_str(), all_word_maps[k][i].c_str());
                        if(temp_p > prob[k][i][j]){
                            prob[k][i][j] = temp_p;
                            last_pos[k][i][j] = q;
                        }
                    }
                }
            }
        }

        // backtrack
        stack<int> result_st;
        int last1 = 0;          //這用來記錄第k個注音要選哪個字
        int last2 = 0;          //這用來記錄第k-1個注音要選哪個字
        double temp_p = prob[row-1][0][0];
        for(int i=0; i<all_word_maps[row-2].size(); i++){       //row-2為除了</s>這個字以外的index，也就是本來句子的最後一字
            if(prob[row-1][0][i] > temp_p){
                temp_p = prob[row-1][0][i];
                last1 = i;
            }
        }
        result_st.push(last1);
        result_st.push(last_pos[row-1][0][last1]);
        last2 = last_pos[row-1][0][last1];

        for(int k=row-2; k>1; k--){
            result_st.push(last_pos[k][last1][last2]);
            last1 = last2;
            last2 = last_pos[k][last1][last2];
        }
        
        // 寫入文字檔
        int i=0;
        outfile << "<s> ";
        while(!result_st.empty()){
            outfile << all_word_maps[i++][result_st.top()] << " " ;
            result_st.pop();
        }
        outfile << "</s>" <<endl;
        //==========================
        delete_3D_Array(last_pos, row, col, col);
        delete_3D_Array_2(prob, row, col, col);
    }

    ex_file.close();
    map_file.close();
    outfile.close();
    return 0;
}