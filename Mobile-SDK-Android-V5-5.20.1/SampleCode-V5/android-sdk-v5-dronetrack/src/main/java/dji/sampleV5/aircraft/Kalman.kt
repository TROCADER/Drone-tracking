package dji.sampleV5.aircraft

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
import kotlin.math.pow

class Kalman {
    private var matrixX: NDArray<Double, D1>? = null
    private var matrixF: NDArray<Double, D2>? = null
    private var matrixH: NDArray<Double, D2>? = null
    private var matrixP: Double? = null
    private var matrixR: NDArray<Double, D2>? = null
    private var matrixQ: NDArray<Double, D2>? = null

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
    // https://filterpy.readthedocs.io/en/latest/_modules/filterpy/common/discretization.html#Q_discrete_white_noise
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

    /**
     * Helper function replicating FilterPy's `order_by_derivative`.
     * Reorders the block diagonal matrix by interleaving dimensions.
     */
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
        stateUncertainty: Double,
        measurementNoise: Double,
        processNoise: Double,
        dt: Double
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

        this.matrixP = stateUncertainty

        this.matrixR = blockDiag(
            mk.ndarray(mk[measurementNoise]),
            mk.ndarray(mk[measurementNoise]),
            mk.ndarray(mk[3 * measurementNoise]),
            mk.ndarray(mk[3 * measurementNoise])
        )

        val q1d = qDiscreteWhiteNoise(2, dt, processNoise, orderByDim = true)
        this.matrixQ = blockDiag(q1d, q1d, q1d, q1d)
    }
}