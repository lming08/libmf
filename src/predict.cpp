#include <iomanip>
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

bool predict(std::string const test_path, std::string const model_path,
             std::string const output_path)
{
    FILE *f = fopen(output_path.c_str(), "w");
    if(!f)
    {
        fprintf(stderr, "\nError: Cannot open %s.", output_path.c_str());
        return false;
    }

    std::shared_ptr<Model> model = read_model(model_path);
    if(!model)
        return false;

    std::shared_ptr<Matrix> Te = read_matrix(test_path);
    if(!Te)
        return false;

    Timer timer;
    timer.tic("Predicting...");

    double loss = 0;
    for(long r = 0; r < Te->nr_ratings; r++)
    {
        float const rate = calc_rate(*model, Te->R[r]);
        fprintf(f, "%f\n", rate);
        float const e = Te->R[r].rate - rate;
        loss += e*e;
    }
    timer.toc("done.");

    printf("RMSE: %.3f\n", sqrt(loss/Te->nr_ratings));
    fclose(f);

    return true;
}
/*
//add qinyuqing function to compute similarity
int  similarity(int const argc, char const * const * const argv){
    std::shared_ptr<PredictOption> option = parse_predict_option(argc, argv);
    if(!option)
        return EXIT_FAILURE;
    FILE *f = fopen(option->out_path.c_str(),"w");
    if(!f){
        fprintf(stderr, "\nError: Cannot open %s.", option->out_path.c_str());
        return false;
    }
    std::shared_ptr<Model> model = read_model(option->model_path);
    if(!model)
        return false;

    Timer timer;
    timer.tic("computing similarity....");
    // compute process
    std::map<float,std::set<int> > container;
    int const dim_aligned = get_aligned_dim(model->param.dim);
    for(int r = 0;r< model->nr_items;r++)
    {
        fprintf(f,"--%d\n",r);
        container.clear();//clear the container
        for(int it=0;it< model->nr_items;it++){
            if(it==r)
                continue;// should't compare self
            float temp = euclidean(model->Q+r*dim_aligned,model->Q+model->param.dim+r*dim_aligned,model->Q+it*dim_aligned);
            if(container.count(temp)){
                container[temp].insert(it);
            }else{
                std::set<int> pid;
                pid.insert(it);
                container.insert(std::make_pair(temp,pid));
            }
            if(container.size()>50)
                container.erase(container.begin());
        }
        //将计算的结果写到文件之中去
        writeSimilarity(f,container);   
    }

    fclose(f);
    return 1;
}

void writeSimilarity(FILE*f,std::map<float,std::set<int> >container){
    std:: map<float,std::set<int>>::iterator it = container.begin();
    while(it!=container.end()){
        fprintf(f,"%f:",it->first);
        std:: set<int>::iterator it2 = it->second.begin();
        while(it2!=it->second.end())
            fprintf(f,"%d;",*it2);

        fprintf(f,"\n");
    }
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
}*/

} //namespace

int predict(int const argc, char const * const * const argv)
{
    std::shared_ptr<PredictOption> option = parse_predict_option(argc, argv);
    if(!option)
        return EXIT_FAILURE;

    if(!predict(option->test_path, option->model_path, option->out_path))
        return EXIT_FAILURE;

    return EXIT_SUCCESS;
}
