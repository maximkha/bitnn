// BitNN
// maximkha

#include <string>   // std::string
#include <bitset>   // std::bitset
#include <cstdlib> 
#include <cstdint>

#define UPDATE_IN_BACKW
typedef int8_t signed_weight_t;
// typedef int_fast8_t signed_weight_t;

typedef uint8_t unsigned_weight_t;
// typedef uint_fast8_t unsigned_weight_t;

typedef size_t accum_size_t;
// typedef uint_fast8_t accum_size_t;

// https://codereview.stackexchange.com/questions/115869/saturated-signed-addition
int saturating_add(signed_weight_t x , signed_weight_t y) { 
    signed_weight_t sum = (unsigned_weight_t) x + y;
    signed_weight_t w = (sizeof(signed_weight_t) << 3) -1;
    signed_weight_t mask = (~(x ^ y) & (x ^ sum)) >> w;
    signed_weight_t max_min = (1 << w) ^ (sum >> w);
    return  (~mask & sum) + (mask & max_min);
}

template <size_t BIT_COUNT>
std::bitset<BIT_COUNT> posBits(signed_weight_t arr[BIT_COUNT])
{
    std::bitset<BIT_COUNT> bitset;
    for (size_t i = 0; i < BIT_COUNT; i++)
    {
        // bitset.set(i, arr[i] > 0);
        bitset[i] = arr[i] > 0;
    }
    return bitset;
}

template <size_t MAT_WIDTH, size_t MAT_HEIGHT>
std::bitset<MAT_HEIGHT> matStepBitVecMul(std::bitset<MAT_WIDTH> mat[MAT_HEIGHT], std::bitset<MAT_WIDTH> vec)
{
    std::bitset<MAT_HEIGHT> bitset;

    for (size_t i = 0; i < MAT_HEIGHT; i++)
    {
        accum_size_t pos_count = (vec & mat[i]).count();
        accum_size_t neg_count = (vec & (~mat[i])).count();
        
        bitset[i] = pos_count > neg_count;
    }
    
    return bitset;
}

template <size_t MAT_WIDTH, size_t MAT_HEIGHT>
std::bitset<MAT_HEIGHT> matStepBitVecMulBias(std::bitset<MAT_WIDTH> mat[MAT_HEIGHT], std::bitset<MAT_HEIGHT>& bias, std::bitset<MAT_WIDTH>& vec)
{
    std::bitset<MAT_HEIGHT> bitset;

    for (size_t i = 0; i < MAT_HEIGHT; i++)
    {
        accum_size_t pos_count = (vec & mat[i]).count() + (bias[i] ? 1 : 0);
        accum_size_t neg_count = (vec & (~mat[i])).count() + (bias[i] ? 0 : 1);

        bitset[i] = pos_count > neg_count;
    }
    
    return bitset;
}

template <size_t MAT_WIDTH, size_t MAT_HEIGHT>
void toBinMat(signed_weight_t arr[MAT_HEIGHT][MAT_WIDTH], std::bitset<MAT_WIDTH> out[MAT_HEIGHT])
{
    for (size_t i = 0; i < MAT_HEIGHT; i++)
    {
        out[i] = posBits<MAT_WIDTH>(arr[i]);
    }
}

template <size_t MAT_WIDTH, size_t MAT_HEIGHT>
void random2DInt8Mat(signed_weight_t mat[MAT_HEIGHT][MAT_WIDTH])
{
    for (size_t i = 0; i < MAT_HEIGHT; i++)
    {
        for (size_t j = 0; j < MAT_WIDTH; j++)
        {
            mat[i][j] = rand() & 0x00ff;
        }
    }
}

template <size_t MAT_WIDTH>
void random1DInt8Mat(signed_weight_t mat[MAT_WIDTH])
{
    for (size_t i = 0; i < MAT_WIDTH; i++)
    {
        mat[i] = rand() & 0x00ff;
    }
}

#pragma region BROKEN_BBSMV
// TODO: fix this
/* 
template <size_t MAT_WIDTH, size_t MAT_HEIGHT>
void backwardBitStepMV(std::bitset<MAT_WIDTH> bmat[MAT_HEIGHT], std::bitset<MAT_WIDTH> inputs, std::bitset<MAT_HEIGHT> grad_zero, std::bitset<MAT_HEIGHT> grad_sign, int8_t raw_mat[MAT_HEIGHT][MAT_WIDTH], std::bitset<MAT_WIDTH> new_grad_zero, std::bitset<MAT_WIDTH> new_grad_sign)
{
    // dloss/din = upstream_loss @ sign(mat.T)
    // dloss/dweights = last_xs.T @ upstream_loss

    // maybe change the accumulator size
    size_t transposedMatMulPositive[MAT_WIDTH];
    size_t transposedMatMulNegative[MAT_WIDTH];

    for (size_t i = 0; i < MAT_HEIGHT; i++)
    {
        // dloss/dweights
        if (grad_zero[i]) continue;

        bool cgradsign = grad_sign[i];

        //Interleaved matmul code for dloss/din

        // To not do two different array iterations, and a transpose,
        // the backprop *won't use* popcount, because we have to popcount 
        // across the transposed version of the weight mat

        // TODO: check if this is more performant to do seperately or in the loop?
        // ~cgradsign, because multiplying by a positive should not change the sign
        std::bitset<MAT_WIDTH> matrow = bmat[i];

        for (size_t j = 0; j < MAT_WIDTH; j++)
        {
            if (~inputs[j]) continue;
            raw_mat[i][j] = saturating_add(raw_mat[i][j], cgradsign ? 1 : -1);
            
            if (matrow[j] ^ (~cgradsign)) transposedMatMulPositive[j]++;
            else transposedMatMulNegative[j]++;
        }
    }

    for (size_t j = 0; j < MAT_WIDTH; j++)
    {
        bool isZero = transposedMatMulPositive[j] == transposedMatMulNegative[j];
        grad_zero[j] = isZero;
        // NOTE: we don't have to update the grad_sign because as long as the entry is marked
        // as a zero, the above grad calculations don't care what the sign is.

        // if (!isZero) {
        //     grad_sign = transposedMatMulPositive[j] > transposedMatMulNegative[j];
        // }

        // TODO: check if the if above is faster or slower
        grad_sign[j] = transposedMatMulPositive[j] > transposedMatMulNegative[j];
    }
}
*/
#pragma endregion BROKEN_BBSMV

template <size_t MAT_WIDTH, size_t MAT_HEIGHT>
void backwardBitStepMVBias(std::bitset<MAT_WIDTH> bmat[MAT_HEIGHT], std::bitset<MAT_HEIGHT>& bbias, std::bitset<MAT_WIDTH>& inputs, std::bitset<MAT_HEIGHT>& grad_zero, std::bitset<MAT_HEIGHT>& grad_sign, signed_weight_t raw_mat[MAT_HEIGHT][MAT_WIDTH], signed_weight_t raw_bias[MAT_HEIGHT], std::bitset<MAT_WIDTH>& new_grad_zero, std::bitset<MAT_WIDTH>& new_grad_sign)
{
    // dloss/din = upstream_loss @ sign(mat.T)
    // dloss/dweights = last_xs.T @ upstream_loss
    
    // A bias essentially acts like a constant 1 input
    // dloss/dbias = [1] @ upstream_loss

    // maybe change the accumulator size
    accum_size_t transposedMatMulPositive[MAT_WIDTH];
    accum_size_t transposedMatMulNegative[MAT_WIDTH];

    // So painful :(
    for (size_t i = 0; i < MAT_WIDTH; i++)
    {
        transposedMatMulNegative[i] = 0;
        transposedMatMulPositive[i] = 0;
    }

    for (size_t i = 0; i < MAT_HEIGHT; i++)
    {
        // dloss/dweights
        // std::cout << "AYO " << "grad_zero= " << grad_zero << " inputs= " << inputs << "\n";
        if (grad_zero[i]) continue;

        bool cgradsign = grad_sign[i];

        //Interleaved matmul code for dloss/din

        // To not do two different array iterations, and a transpose,
        // the backprop *won't use* popcount, because we have to popcount 
        // across the transposed version of the weight mat

        // TODO: check if there's a better way of doing this
        // ~cgradsign, because multiplying by a positive should not change the sign
        std::bitset<MAT_WIDTH> matrow = bmat[i];

        for (size_t j = 0; j < MAT_WIDTH; j++)
        {
            // Irregardless of input

            if (matrow[j] != (!cgradsign)) transposedMatMulPositive[j]++;
            else transposedMatMulNegative[j]++;

            if (~inputs[j]) continue;

            // update bit mat
            #ifdef UPDATE_IN_BACKW
            // crosses bound to positive: (raw_mat[i][j] == 0) && !cgradsign;
            if ((raw_mat[i][j] == 0) && !cgradsign) bmat[i][j] = true;
            // crosses bound to negative: (raw_mat[i][j] == 1) && cgradsign;
            if ((raw_mat[i][j] == 1) && cgradsign) bmat[i][j] = false;
            #endif

            raw_mat[i][j] = saturating_add(raw_mat[i][j], cgradsign ? -1 : 1);
        }

        // update bit mat (bias)
        #ifdef UPDATE_IN_BACKW
        if ((raw_bias[i] == 0) && !cgradsign) bbias[i] = true;
        if ((raw_bias[i] == 1) && cgradsign) bbias[i] = false;
        #endif

        raw_bias[i] = saturating_add(raw_bias[i], cgradsign ? -1 : 1);
    }

    for (size_t j = 0; j < MAT_WIDTH; j++)
    {
        bool isZero = transposedMatMulPositive[j] == transposedMatMulNegative[j];
        new_grad_zero[j] = isZero;
        // NOTE: we don't have to update the grad_sign because as long as the entry is marked
        // as a zero, the above grad calculations don't care what the sign is.

        // if (!isZero) {
        //     grad_sign = transposedMatMulPositive[j] > transposedMatMulNegative[j];
        // }

        // TODO: check if the if above is faster or slower
        new_grad_sign[j] = transposedMatMulPositive[j] > transposedMatMulNegative[j];
    }
}