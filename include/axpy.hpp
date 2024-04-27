#ifndef AXPY_HPP
#define AXPY_HPP 


template<typename T>
class Axpy {
private:
    // Private member variables and methods
    T * d_x_;
    T * d_y_;
    int N_;
    int device_id_;

public:
    // Constructors
    Axpy(int N, int deviceID=0);

    // Destructor
    ~Axpy();

    void Load(T * h_x, T * h_y);
    void Execute(T a);
    void Purge(T * h_y);
};



#endif // AXPY_HPP
