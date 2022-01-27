#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <vector>
using namespace std;

double e = 1e-16; //终止温度
double at = 0.9; //温度变化率
double T = 1.0; //初始温度
int L = 2000; //最大迭代次数
int main()
{
    freopen("data.txt", "r", stdin);
    freopen("out.txt", "w", stdout);
    int L; //area length
    int W; //area width
    int M; //number of cell
    int N; //number of net
    cin>>L;
    cin>>W; 
    cin>>M;
    cin>>N;
    vector<vector<int>> net(N);
    getchar();
    getchar();
    int a;
    //input data
    for(int i = 0;i < N; i++){
        cin>>a;
        //cout<<a;
        net[i].push_back(a);
        while (cin.get() != '\n') 
        {
            cin >> a;
            net[i].push_back(a);
        }
        //cout<<"\n";
    }
    //初始化数据，为了尽量减少
    int **place;//placement 范围
    place = ( int  **)  malloc ( sizeof ( int  *) *L); //申请一组一维指针空间。
    for (int i = 0; i < L; i ++)
        place[i] = ( int  *) malloc ( sizeof ( int ) * W);  //对于每个一维指针，申请一行数据的空间。

    
    return 0;
}