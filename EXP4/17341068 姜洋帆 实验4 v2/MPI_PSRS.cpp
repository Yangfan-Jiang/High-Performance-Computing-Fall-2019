#include<iostream>
#include<fstream>
#include<cstdlib>
#include<algorithm>
#include<string>
#include<cmath>
#include<mpi.h>

using namespace std;

int main() {
    int comm_sz, my_rank;
    double start, end;
    bool ordered;  // if local data correct
    
    /* get the total number of element to sort */
    unsigned long total_num;
    unsigned long check_total_num = 0;

    unsigned long preMaxData;
    unsigned long currMinData;

    unsigned long local_start;
    unsigned long local_length;

    char filepath[100] = "/public/home/shared_dir/psrs_data";
    //char filepath[100] = "./data";

    ifstream fin(filepath, ios::binary);
    fin.read((char*)&total_num, 8); 
    
    /* MPI initialization */
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    local_start = my_rank*floor(total_num/comm_sz) + 1;
    local_length = total_num/comm_sz;

    /* read local data for each process */
    if(my_rank == comm_sz-1) {
        local_length = total_num - local_start + 1;
    }
    
    fin.seekg((local_start)*8, ios::beg);
    
    unsigned long *local_data = new unsigned long[local_length];
    for(unsigned long i=0; i<local_length; i++) {
        fin.read((char*)&local_data[i], 8);
    }
    fin.close();
    
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    
    /* sort the local data for each process */
    sort(local_data, local_data+local_length);
    
    /* local regular samples */
    unsigned long *regularSamples = new unsigned long[comm_sz];
    for(int i=0; i<comm_sz; i++) {
        regularSamples[i] = local_data[i*local_length/comm_sz];
    }

    /* root process recive all regular samples and sort it */
    unsigned long *RegSam;
    if(my_rank == 0) {
        RegSam = new unsigned long[comm_sz*comm_sz];
    }
    MPI_Gather(regularSamples, comm_sz, MPI_UNSIGNED_LONG, RegSam, comm_sz, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // root process sort global RegSam
    unsigned long *privots= new unsigned long[comm_sz-1];
    if(my_rank == 0) {
        sort(RegSam, RegSam+comm_sz*comm_sz);
        for(int i=0; i<comm_sz-1; i++) {
            privots[i] = RegSam[(i+1)*comm_sz];
        }
    }

    /* bcast privots to all processes */
    MPI_Bcast(privots, comm_sz-1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    /* partition local data by privots */
    int privotsIndex = 0;
    int *partitionIndex = new int[comm_sz];
    int *partitionLen = new int[comm_sz];
    partitionIndex[0] = 0;
    unsigned long partition_len = 0;

    for(unsigned long i=0; i<local_length; i++) {
        partition_len++;
        if(local_data[i] <= privots[privotsIndex] && privots[privotsIndex] <= local_data[i+1]) {
            partitionLen[privotsIndex] = partition_len;
            partitionIndex[privotsIndex+1] = i+1;
            partition_len = 0;
            privotsIndex++;
        }
        if(privotsIndex == comm_sz-1) {
            partitionLen[privotsIndex] = local_length - i - 1;
            break;
        }
        if(i == local_length-1) {
            partitionLen[privotsIndex] = partition_len;
            for(int index=privotsIndex+1; index<comm_sz; index++)
                partitionLen[index]=0;
        }
    }

    /* All to all operation */
    // 1. send length of each partition
    int *ParLen = new int[comm_sz];
    MPI_Alltoall(partitionLen, 1, MPI_INT, ParLen, 1, MPI_INT, MPI_COMM_WORLD);

    // 2. calculate total length of partition for each process
    int new_length = 0;
    int *new_parIndex = new int[comm_sz];
    for(int i=0; i<comm_sz; i++) {
        new_parIndex[i] = new_length;
        new_length += ParLen[i];
    }
    unsigned long s_new_length = long(new_length);

    // 3. receive partitions (data)
    unsigned long *recvPartition = new unsigned long[new_length];
    MPI_Alltoallv(local_data, partitionLen, partitionIndex, MPI_UNSIGNED_LONG, 
                recvPartition, ParLen, new_parIndex, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    
    MPI_Reduce(&s_new_length, &check_total_num, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if(my_rank == 0) {
        if(check_total_num != total_num) {
            cout << check_total_num << endl;
            cout << "Error: total number of element incorrect!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 911);
        }
    }

    // 4. merge sorted partitions
    // using quick sort for test
    sort(recvPartition, recvPartition+new_length);
    
    for(int j=0; j<new_length-1;j++){
        if(recvPartition[j] > recvPartition[j+1]) {
            cout << "error: rank " << my_rank << " local data incorrect" << endl;
            MPI_Abort(MPI_COMM_WORLD, 911);
        }
    }

    // check global data    
    if(my_rank == 0)
        preMaxData = -1;

    if(my_rank % 2 == 0) {
        MPI_Send(&recvPartition[new_length-1], 1, MPI_UNSIGNED_LONG, my_rank+1, 0, MPI_COMM_WORLD);
        if(my_rank != 0)
            MPI_Recv(&preMaxData, 1, MPI_UNSIGNED_LONG, my_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        MPI_Recv(&preMaxData, 1, MPI_UNSIGNED_LONG, my_rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(my_rank != comm_sz-1)
            MPI_Send(&recvPartition[new_length-1], 1, MPI_UNSIGNED_LONG, my_rank+1, 0, MPI_COMM_WORLD);
    }

    if(my_rank != 0 && (preMaxData > recvPartition[0])) {
        cout << "error: global data incorrect!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 911);
    } 

    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();
    MPI_Finalize();

    if(my_rank == 0) {
        cout << "PSRS run successfully!" << endl; 
        cout << "elapsed: "<< (end - start)*1000 << "ms with " << comm_sz << " processes" << endl;
    }
    return 0;
}