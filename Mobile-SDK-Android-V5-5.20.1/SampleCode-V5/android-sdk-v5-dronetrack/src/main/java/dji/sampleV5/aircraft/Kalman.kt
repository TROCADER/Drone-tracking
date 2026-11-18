package dji.sampleV5.aircraft

import dji.sdk.keyvalue.value.common.RectF
import org.jetbrains.kotlinx.multik.api.identity
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.linalg.inv
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1
import org.jetbrains.kotlinx.multik.ndarray.data.D2
import org.jetbrains.kotlinx.multik.ndarray.data.DataType
import org.jetbrains.kotlinx.multik.ndarray.data.Dimension
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray
import org.jetbrains.kotlinx.multik.ndarray.data.NDArray
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.map
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow


class Kalman {
    var matrixX: NDArray<Double, D1> = mk.zeros(8)   // State vector x (8x1)
    private var matrixF: NDArray<Double, D2> = mk.zeros(8,8)   // State transition matrix F (8x8)
    private var matrixH: NDArray<Double, D2> = mk.zeros(4,8)   // Measurement matrix H (4x8)
    private var matrixN: NDArray<Double, D2> = mk.zeros(8,8)   // Process noise input matrix N (8x8)
    private var matrixR1: NDArray<Double, D2> = mk.zeros(8,8)  // Process noise covariance matrix R1 (8x8)
    private var matrixR2: NDArray<Double, D2> = mk.zeros(8,8)  // Measurement noise covariance matrix R2 (4x4)
    private var matrixR12: NDArray<Double, D2> = mk.zeros(8,4) // Cross-covariance matrix R12 (8x4)
    private var matrixP: NDArray<Double, D2> =  mk.zeros(8,8)   // Error covariance matrix P (8x8)
    private var matrixK: NDArray<Double, D2> = mk.zeros(8,4)// Kalman gain matrix K (8x4)

    constructor(
        rect: RectF,
        stateUncertainty: Double = 1.0,
        measurementNoise: Double = 0.1,
        processNoise: Double = 0.01,
        dt: Double = 0.1
    ) {
        val l = rect.left ?: 0.0
        val t = rect.top ?: 0.0
        val r = rect.right ?: 0.0
        val b = rect.bottom ?: 0.0
        val pos = mk.ndarray(mk[l, t, r, b])
        createKalman(pos, stateUncertainty, measurementNoise, processNoise, dt)
    }


    var missedFrames: Int = 0
        private set

    private val maxMissedFrames: Int = 15

    // Translated from scipy.linalg.block_diag
    // https://github.com/scipy/scipy/blob/v1.18.0/scipy/linalg/_special_matrices.py#L361-L451
    fun <T : Number> blockDiag(
        vararg arrs: MultiArray<T, out Dimension>,
        dtype: DataType = DataType.DoubleDataType
    ): NDArray<T, D2> {

        // If no arrays are provided, mimic SciPy's behavior: return a (1, 0) matrix
        if (arrs.isEmpty()) {
            return mk.zeros(intArrayOf(1, 0), dtype)
        }

        // Determine the final matrix dimensions
        var totalRows = 0
        var totalCols = 0

        for (arr in arrs) {
            when (arr.dim.d) {
                1 -> {
                    totalRows += 1
                    totalCols += arr.shape[0]
                }

                2 -> {
                    totalRows += arr.shape[0]
                    totalCols += arr.shape[1]
                }

                else -> throw IllegalArgumentException("blockDiag currently only supports 1D and 2D arrays.")
            }
        }

        // Allocate the output array filled with zeros
        val out = mk.zeros<T, D2>(intArrayOf(totalRows, totalCols), dtype)

        // Fill the diagonal blocks
        var r = 0
        var c = 0

        for (mat in arrs) {
            val rows = if (mat.dim.d == 1) 1 else mat.shape[0]
            val cols = if (mat.dim.d == 1) mat.shape[0] else mat.shape[1]

            if (mat.dim.d == 1) {
                @Suppress("UNCHECKED_CAST")
                val d1Mat = mat as NDArray<T, D1>
                for (j in 0 until cols) {
                    out[r, c + j] = d1Mat[j]
                }
            } else {
                @Suppress("UNCHECKED_CAST")
                val d2Mat = mat as NDArray<T, D2>
                for (i in 0 until rows) {
                    for (j in 0 until cols) {
                        out[r + i, c + j] = d2Mat[i, j]
                    }
                }
            }

            r += rows
            c += cols
        }

        return out
    }

    // Translated from filterpy.common.discretization.Q_discrete_white_noise
    // https://github.com/rlabbe/filterpy/blob/master/filterpy/common/discretization.py#L70-L145
    fun qDiscreteWhiteNoise(
        dim: Int,
        dt: Double = 1.0,
        variance: Double = 1.0,
        blockSize: Int = 1,
        orderByDim: Boolean = true
    ): NDArray<Double, D2> {

        if (dim !in 2..4) {
            throw IllegalArgumentException("dim must be between 2 and 4")
        }

        // Build the core Q matrix based on dimension
        val qMatrix: NDArray<Double, D2> = when (dim) {
            2 -> {
                mk.ndarray(
                    doubleArrayOf(
                        0.25 * dt.pow(4), 0.5 * dt.pow(3),
                        0.5 * dt.pow(3), dt.pow(2)
                    )
                ).reshape(2, 2)
            }

            3 -> {
                mk.ndarray(
                    doubleArrayOf(
                        0.25 * dt.pow(4), 0.5 * dt.pow(3), 0.5 * dt.pow(2),
                        0.5 * dt.pow(3), dt.pow(2), dt,
                        0.5 * dt.pow(2), dt, 1.0
                    )
                ).reshape(3, 3)
            }

            4 -> {
                mk.ndarray(
                    doubleArrayOf(
                        dt.pow(6) / 36.0, dt.pow(5) / 12.0, dt.pow(4) / 6.0, dt.pow(3) / 6.0,
                        dt.pow(5) / 12.0, dt.pow(4) / 4.0, dt.pow(3) / 2.0, dt.pow(2) / 2.0,
                        dt.pow(4) / 6.0, dt.pow(3) / 2.0, dt.pow(2), dt,
                        dt.pow(3) / 6.0, dt.pow(2) / 2.0, dt, 1.0
                    )
                ).reshape(4, 4)
            }

            else -> throw IllegalStateException("Unreachable")
        }

        val finalMatrix = if (orderByDim) {
            // Create an array of `blockSize` duplicates of `qMatrix`
            val qBlocks = Array(blockSize) { qMatrix }
            blockDiag(*qBlocks, dtype = DataType.DoubleDataType)
        } else {
            orderByDerivative(qMatrix, dim, blockSize)
        }

        // Multiply the final result by the variance
        return finalMatrix.map { it * variance }
    }

    fun orderByDerivative(q: NDArray<Double, D2>, dim: Int, blockSize: Int): NDArray<Double, D2> {
        val n = dim * blockSize
        val d = mk.zeros<Double, D2>(intArrayOf(n, n), DataType.DoubleDataType)

        for (r in 0 until dim) {
            for (c in 0 until dim) {
                val x = q[r, c]
                for (j in 0 until blockSize) {
                    d[r * blockSize + j, c * blockSize + j] = x
                }
            }
        }

        return d
    }

    fun createKalman(
        initPos: NDArray<Double, D1>,
        stateUncertainty: Double = 1.0,
        measurementNoise: Double = 0.1,
        processNoise: Double = 0.01,
        dt: Double = 0.1
    ) {
        this.matrixX = mk.ndarray(
            mk[initPos[0], 0.0, initPos[1], 0.0, initPos[2], 0.0, initPos[3], 0.0]
        )

        this.matrixF = mk.ndarray(
            mk[
                mk[1.0, dt, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                mk[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                mk[0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 0.0],
                mk[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                mk[0.0, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0],
                mk[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                mk[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, dt],
                mk[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            ]
        )

        this.matrixH = mk.ndarray(
            mk[
                mk[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                mk[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                mk[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                mk[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ]
        )

        this.matrixN = mk.identity<Double>(8)

        val q1d = qDiscreteWhiteNoise(2, dt, processNoise, orderByDim = true)
        this.matrixR1 = blockDiag(q1d, q1d, q1d, q1d)

        this.matrixR2 = blockDiag(
            mk.ndarray(mk[measurementNoise]),
            mk.ndarray(mk[measurementNoise]),
            mk.ndarray(mk[3 * measurementNoise]),
            mk.ndarray(mk[3 * measurementNoise])
        )

        this.matrixR12 = mk.zeros<Double, D2>(intArrayOf(8, 4), DataType.DoubleDataType)

        this.matrixP = mk.identity<Double>(8).map { it * stateUncertainty }

        this.isInitialized = true
        this.missedFrames = 0
    }

// K = (F * P * H^T + N * R12) * (H * P * H^T + R2)^-1
// x_next = F * x + K * (z - H * x)
// P_next = F * P * F^T + N * R1 * N^T - (F * P * H^T + N * R12) * (H * P * H^T + R2)^-1 * (F * P * H^T + N * R12)^T
    fun update(z: NDArray<Double, D1>) {
        val x = matrixX ?: return
        val F = matrixF ?: return
        val H = matrixH ?: return
        val N = matrixN ?: return
        val R1 = matrixR1 ?: return
        val R2 = matrixR2 ?: return
        val R12 = matrixR12 ?: return
        val P = matrixP ?: return

        // Innovation / residual: y = z - H * x
        val Hx = H dot x
        val y = z - Hx

        // (F * P * H^T + N * R12)
        val HT = H.transpose()
        val FP = mk.linalg.dot(F, P)
        val FPH_T = mk.linalg.dot(FP, HT)
        val NR12 = mk.linalg.dot(N, R12)
        val termFPH_NR12 = FPH_T + NR12

        // S = (H * P * H^T + R2)
        val HP = mk.linalg.dot(H, P)
        val HPH_T = mk.linalg.dot(HP, HT)
        val S = HPH_T + R2
        val SInv = mk.linalg.inv(S)

        // K = (F * P * H^T + N * R12) * (H * P * H^T + R2)^-1
        val K = mk.linalg.dot(termFPH_NR12, SInv)
        this.matrixK = K

        // x_next = F * x + K * y
        val Fx = F dot x
        val Ky = K dot y
        val xNext = Fx + Ky
        this.matrixX = xNext

        // P_next = F * P * F^T + N * R1 * N^T - K * (F * P * H^T + N * R12)^T
        val FT = F.transpose()
        val FPF_T = mk.linalg.dot(FP, FT)

        val NT = N.transpose()
        val NR1 = mk.linalg.dot(N, R1)
        val NR1N_T = mk.linalg.dot(NR1, NT)

        val termFPF_NR1N = FPF_T + NR1N_T
        val termFPH_NR12_T = termFPH_NR12.transpose()
        val K_termT = mk.linalg.dot(K, termFPH_NR12_T)

        val PNext = termFPF_NR1N - K_termT
        this.matrixP = PNext
    }

//    x_next = F * x
//    P_next = F * P * F^T + N * R1 * N^T
    fun predict() {
        val x = matrixX ?: return
        val F = matrixF ?: return
        val N = matrixN ?: return
        val R1 = matrixR1 ?: return
        val P = matrixP ?: return

        this.matrixX = F dot x

        val FP = mk.linalg.dot(F, P)
        val FT = F.transpose()
        val FPF_T = mk.linalg.dot(FP, FT)

        val NT = N.transpose()
        val NR1 = mk.linalg.dot(N, R1)
        val NR1N_T = mk.linalg.dot(NR1, NT)

        this.matrixP = FPF_T + NR1N_T
    }

    fun currentEstimateRect(): RectF {
        val x = matrixX
        val top =    x[0] ?: 0.0
        val left =   x[2] ?: 0.0
        val width =  x[4] ?: 0.0
        val height = x[6] ?: 0.0

        return RectF(left, top, left + width, top + height)
    }
    /*
    fun reset() {
        matrixX = null
        matrixF = null
        matrixH = null
        matrixN = null
        matrixR1 = null
        matrixR2 = null
        matrixR12 = null
        matrixP = null
        matrixK = null
        missedFrames = 0
    }*/
}
