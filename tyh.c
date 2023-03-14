#include <g2o/core/block_solver.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_edge.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv/cv.hpp>
#include <chrono>

using namespace std;
using namespace g2o;
using namespace Eigen;

//定义顶点
class CurveFittingVertex:public BaseVertex<3,Vector3d>{//3：雅克比矩阵纬度
  public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW//开启内存对齐
  //重置
  virtual void setToOriginImpl() override{
    _estimate<<0,0,0;//_estimate是BaseVertex内置成员变量
  }
  //更新
  virtual void oplusImpl(const double *update)override{
    _estimate+=Vector3d(update);
  }

  //读盘留空
  virtual bool read(istream &in){}
  virtual bool write(ostream &out)const{}
  //存盘留空
};

//定义边
class CurveFittingEdge:public BaseUnaryEdge<1,double,CurveFittingVertex>{//1：误差项的维度
  public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW//开启内存对齐
   CurveFittingEdge(double x):BaseUnaryEdge(),_x(x){};

    //计算曲线模型误差
    virtual void computeError() override{
      const CurveFittingVertex *v=static_cast<const CurveFittingVertex *> (_vertices[0]);//读取顶点信息
      const Vector3d abc=v->estimate();//返回对当前顶点姿态的估计
      
      _error(0,0)=_measurement-exp(abc(0,0)*_x*_x+abc(1,0)*_x+abc(2,0));//误差计算
    }

    //计算雅克比矩阵
    virtual void linearizeOplus() override{
      const CurveFittingVertex *v=static_cast<const CurveFittingVertex *>(_vertices[0]);
      const Vector3d abc=v->estimate();
      double y=exp(abc[0]*_x*_x+abc[1]*_x+abc[2]);//观测量
      _jacobianOplusXi[0]=-_x*_x*y;//求一阶导
      _jacobianOplusXi[1]=-_x*y;
      _jacobianOplusXi[2]=-y;
    }

    virtual bool read(istream &in){}
    virtual bool write(ostream &out)const{}
    public:
      double _x;//x值，y 值为 _measurement
};
int main(int argc, char const *argv[])
{
  double ar=1,br=2,cr=1;   //真实值
  double ae=2,be=0,ce=5.0;//参照值
  int N=100;              //数据个数
  double w_sigma=1;       //噪声sigma值
  double inv_sigma=1.0/w_sigma;
  cv::RNG rng;
  vector<double> x_data,y_data; //数据
  for(int i=0;i<N;i++){
    double x=i/100;
    x_data.push_back(x);
    y_data.push_back(exp(ar*x*x+br*x+cr)+rng.gaussian(w_sigma*w_sigma));
  }
  
  //构建图优化
  
  typedef BlockSolver<BlockSolverTraits<3,1>> BlockSolverType;
  //BlockSolverTraits对应一条边两个顶点的纬度<3:雅克比矩阵的纬度，1：观测量的纬度>
  typedef LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
  //线性求解器类型

  auto solver=new OptimizationAlgorithmGaussNewton(g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
  //make_unique 调用unique_ptr实现内存地址绑定
  SparseOptimizer optimizer;//构建一个图模型性类
  optimizer.setAlgorithm(solver);//设置求解器
  optimizer.setVerbose(true);//打开调试输出

  //往图中添加顶点
  CurveFittingVertex *v=new CurveFittingVertex();
  v->setEstimate(Vector3d(ae,be,ce));//设置初始值
  v->setId(0);
  optimizer.addVertex(v);

  //往图中添加边
  for(int i=0;i<N;i++){
    CurveFittingEdge *edge=new CurveFittingEdge(x_data[i]);//创建边
    edge->setId(i);                       //设置ID
    edge->setVertex(0, v);                // 设置连接的顶点
    edge->setMeasurement(y_data[i]);      //设置测量值
    edge->setInformation(Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma));
    optimizer.addEdge(edge);              //添加
  }

  //执行优化
  cout << "start optimization" << endl;
  chrono::steady_clock::time_point t1=chrono::steady_clock::now();//获取当前时间
  optimizer.initializeOptimization();//初始化
  optimizer.optimize(20);  //执行10次
  chrono::steady_clock::time_point t2=chrono::steady_clock::now();//获取当前时间
  chrono::duration<double> time_used= chrono::duration_cast<chrono::duration<double>>(t2 - t1);//获取时间间隔
  cout <<"solve time cost :"<<time_used.count()<<endl;

  Vector3d abc_estimate = v->estimate();//获取当前值
  cout<<"estimated model:"<<abc_estimate.transpose()<<endl;
  return 0;
}

