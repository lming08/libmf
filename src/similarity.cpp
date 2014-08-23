#include <iomanip>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "mf.h"

namespace
{

struct PredictOption
{
    std::string test_path, model_path, out_path;
};

void predict_help()
{
    printf("usage: libmf predict binary_test_file model [output]\n");
}

std::shared_ptr<PredictOption> parse_predict_option(
        const int argc, const char * const * const argv)
{
    if((argc != 2) && (argc != 3))
    {
        predict_help();
        return std::shared_ptr<PredictOption>(nullptr);
    }

    std::shared_ptr<PredictOption> option(new PredictOption);

    option->test_path = std::string(argv[0]);
    option->model_path = std::string(argv[1]);
    if(argc == 3)
    {
        option->out_path = std::string(argv[2]);
    }
    else
    {
        const char *p = strrchr(argv[0], '/');
        if(!p)
            p = argv[0];
        else
            ++p;
        option->out_path = std::string(p) + ".out";
    }
    return option;
}

//欧氏距离
float euclidean(float* begin,float* end,float * second){
    float result = 0.0;
    while(begin!=end){
        result+=(*begin-*second)*(*begin-*second);
        begin++;
        second++;
    }
    return sqrt(result);
}

float cosine(float* begin,float* end,float * second){
    float dot = std::inner_product(begin,end,second,0.0);
    float squareSum1 = 0.0;
    float squareSum2 = 0.0;
    while(begin!=end){
        squareSum1+=(*begin)*(*begin);
        squareSum2+=(*second)*(*second);
        begin++;
        second++;
    }
    return dot/(sqrt(squareSum1)*sqrt(squareSum2));
}

float pearson(float*begin,float *end,float* second){
    int len = end-begin;
    float pSum = std::inner_product(begin,end,second,0.0);
    float sum1 = 0.0;
    float sum2 = 0.0;
    float sumSq1 = 0.0;
    float sumSq2 = 0.0;
    while(begin!=end){
        sum1+=*begin;
	sum2+=*second;
	sumSq1+= (*begin)*(*begin);
	sumSq2+= (*second)*(*second);
	begin++;
	second++;
    }
    float fractor = pSum-(sum1*sum2/len);
    float deno = sqrt((sumSq1-pow(sum1,2)/len)*(sumSq2-pow(sum2,2)/len));
    if(deno==0.0)
    	return 0.0;
    else
        return fractor/deno;
}

void writeSimilaritySmall(FILE*f,std::map<float,std::set<int> >& container){
    std:: map<float,std::set<int>>::iterator it = container.begin();
    while(it!=container.end()){
        fprintf(f,"%f:",it->first);
        std:: set<int>::iterator it2 = it->second.begin();
        while(it2!=it->second.end()){
            fprintf(f,"%d;",*it2);
            it2++;
        }
        it++;
        fprintf(f,"|");
    }
    fprintf(f,"\n");
}

void writeSimilarityBigger(FILE*f,std::map<float,std::set<int> >& container){
    std:: map<float,std::set<int>>::reverse_iterator it = container.rbegin();
    while(it!=container.rend()){
        fprintf(f,"%f:",it->first);
        std:: set<int>::iterator it2 = it->second.begin();
        while(it2!=it->second.end()){
            fprintf(f,"%d;",*it2);
            it2++;
        }
        it++;
        fprintf(f,"|");
    }
    fprintf(f,"\n");
}

bool readOnLine(std::string path,std::set<int>& online){
	FILE* f = fopen(path.c_str(),"r");
	if(!f){
	    return false;
	}
	Timer timer;
	timer.tic("read online session");
	while(true){
		int number = -1;
		if(fscanf(f,"%d\n",&number)==EOF)
			break;
		online.insert(number);
	}
	fclose(f);
	printf("online length:%u\n",(unsigned int)online.size());
	return true;
}


//add qinyuqing function to compute similarity
int  similarity(std::string const model_path,std::string const out_path){
    FILE* euc = fopen(out_path.c_str(),"w");
    FILE* pear = fopen("./pearson.txt","w");
    FILE* cos = fopen("./cosine.txt","w");
    FILE * writer = fopen("./userCustomer","w");
    	
    if(!euc){
        fprintf(stderr, "\nError: Cannot open %s.", out_path.c_str());
        return false;
    }
    std::shared_ptr<Model> model = read_model(model_path);
    if(!model)
        return false;

    // compute process
    std::map<float,std::set<int> > eucontainer;
    std::map<float,std::set<int> > coscontainer;
    std::map<float,std::set<int> > pearcontainer;
    int const dim_aligned = get_aligned_dim(model->param.dim);

    //写出text特征向量
    printf("param.dim: %d,get_aligned_dim:%d\n",model->param.dim,dim_aligned);
/*    printf("write the binary feature to file\n");
    FILE *feature_user = fopen("feature_user.txt","w");
    FILE *feature_item = fopen("feature_item.txt","w");
    for(int i=0;i<model->nr_users;i++){
    	for(int j=0;j<model->param.dim;j++){
		fprintf(feature_user,"%f;",*(model->P+i*dim_aligned+j));
	}
	fprintf(feature_user,"\n");
    }

    for(int i=0;i<model->nr_items;i++){
    	for(int j=0;j<model->param.dim;j++){
		fprintf(feature_item,"%f;",*(model->Q+i*dim_aligned+j));
	}
	fprintf(feature_item,"\n");
    }

    fclose(feature_user);
    fclose(feature_item);
*/
    std::set<int> online;
    readOnLine("../online",online);

    //写出给部分用户推荐的商品
    printf("compute  user customer\n");
    std::map<float,std::set<int> > recommendation;
    std::set<int>::iterator it;

    for(int u = 0;u<model->nr_users;u++){
    	recommendation.clear();
	fprintf(writer,"%d\t",u);
	it = online.begin();
	while(it!=online.end()){
	    float rate  = std::inner_product(model->P+u*dim_aligned,
	    	model->P+u*dim_aligned+model->param.dim,model->Q+(*it)*dim_aligned,0.0);
	    if(recommendation.count(rate))
	    	recommendation[rate].insert(*it);
	    else{
	    	std::set<int> pid;
		pid.insert(*it);
	    	recommendation.insert(std::make_pair(rate,pid));
	    }
	    if(recommendation.size()>100)
	    	recommendation.erase(recommendation.begin());

	    it++;
	}
    	writeSimilarityBigger(writer,recommendation);
    }
    recommendation.clear();
    fclose(writer);

/*    
    Timer timer;
    timer.tic("computing similarity....");
    //计算商品之间的相似度
    for(int r = 0;r< model->nr_items;r++)
    {
        fprintf(euc,"%d\t",r);//PID map start 0
        fprintf(cos,"%d\t",r);
	fprintf(pear,"%d\t",r);

        eucontainer.clear();//clear the container
        coscontainer.clear();
	pearcontainer.clear();

        for(int it=0;it< model->nr_items;it++){
            if(it==r)
                continue;// should't compare self
            float tempeu = euclidean(model->Q+r*dim_aligned,
                    model->Q+model->param.dim+r*dim_aligned,model->Q+it*dim_aligned);
            float tempcos = cosine(model->Q+r*dim_aligned,
                    model->Q+model->param.dim+r*dim_aligned,model->Q+it*dim_aligned);
	    float temppear = pearson(model->Q+r*dim_aligned,
	    	    model->Q+model->param.dim+r*dim_aligned,model->Q+it*dim_aligned);
            //eucian distance 
            if(eucontainer.count(tempeu)){
                eucontainer[tempeu].insert(it);
            }else{
                std::set<int> pid;
                pid.insert(it);
                eucontainer.insert(std::make_pair(tempeu,pid));
            }
            if(eucontainer.size()>50)
                eucontainer.erase(--eucontainer.end());

            //cosine distance
            if(coscontainer.count(tempcos)){
                coscontainer[tempcos].insert(it);
            }else{
                std::set<int> pid;
                pid.insert(it);
                coscontainer.insert(std::make_pair(tempcos,pid));
            }
            if(coscontainer.size()>50)
                coscontainer.erase(coscontainer.begin());
	
	    //pearson
	    if(pearcontainer.count(temppear))
	        pearcontainer[temppear].insert(it);
	    else{
	    	std::set<int> pid;
		pid.insert(it);
		pearcontainer.insert(std::make_pair(temppear,pid));
	    }
	    if(pearcontainer.size()>50)
	    	pearcontainer.erase(pearcontainer.begin());
	        
        }
        //将计算的结果写到文件之中去
        writeSimilaritySmall(euc,eucontainer); 
        writeSimilarityBigger(cos,coscontainer);
	writeSimilarityBigger(pear,pearcontainer);
    }
*/
    fclose(euc);
    fclose(cos);
    fclose(pear);	

    return 1;
}


} //namespace

int similarity(int const argc, char const * const * const argv)
{
    std::shared_ptr<PredictOption> option = parse_predict_option(argc, argv);
    if(!option)
        return EXIT_FAILURE;

    similarity(option->model_path,option->out_path);
    return EXIT_SUCCESS;
}
