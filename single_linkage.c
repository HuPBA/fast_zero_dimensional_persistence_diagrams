#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>


void fillDoubleArray(double* array, int size, double value) {
    for (int i = 0; i < size; i++)
        array[i] = value;
}

void fillIntArray(int* array, int size, int value) {
    for (int i = 0; i < size; i++)
        array[i] = value;
}

int condensedIndex(int n, int i, int j) {
    if(i < j)
        return n * i - (i * (i + 1) / 2) + (j - i - 1);
    else if(i > j)
        return n * j - (j * (j + 1) / 2) + (i - j - 1);
    else
        return -1;
}

static PyObject *zero_persistence_diagram_by_single_linkage_algorithm(PyObject* Py_UNUSED(self), PyObject *args) {
    PyObject *condensedMatrixRaw, *condensedMatrix, *persistenceDiagram, *persistenceIndices;
    int numberOfPoints, *merged, *DIndex, *persistenceIndicesData, y=0, x=0, matrixEntries;
    double currentMin, dist, *D, *condensedMatrixData, *persistenceDiagramData;
    // Parsing arguments
    if (!PyArg_ParseTuple(args, "O", &condensedMatrixRaw))
        return NULL;
    // Get condensed matrix
    condensedMatrix = PyArray_FROM_OTF(condensedMatrixRaw, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (condensedMatrix == NULL) {
        Py_XDECREF(condensedMatrix);
        return NULL;
    }
    matrixEntries = (int) PyArray_SIZE((PyArrayObject*)condensedMatrix);
    numberOfPoints = (int) (1 + sqrt(1 + 8 * matrixEntries)) / 2;
    // Allocate memory
    D = (double *) malloc(numberOfPoints * sizeof(double));
    DIndex = (int *) malloc(numberOfPoints * sizeof(int));
    merged = (int *) malloc(numberOfPoints * sizeof(int));
    npy_intp dimsOutput[1] = {numberOfPoints-1};
    persistenceDiagram = PyArray_SimpleNew(1, dimsOutput, NPY_DOUBLE);
    persistenceIndices = PyArray_SimpleNew(1, dimsOutput, NPY_INT);
    if (persistenceDiagram == NULL || persistenceIndices == NULL) {
        free(D);
        free(DIndex);
        free(merged);
        Py_XDECREF(condensedMatrix);
        Py_XDECREF(persistenceDiagram);
        Py_XDECREF(persistenceIndices);
        return NULL;
    }
    persistenceDiagramData = (double *) PyArray_DATA((PyArrayObject*)persistenceDiagram);
    persistenceIndicesData = (int *) PyArray_DATA((PyArrayObject*)persistenceIndices);
    condensedMatrixData = (double *) PyArray_DATA((PyArrayObject*)condensedMatrix);
    // Single linkage algorithm
    fillDoubleArray(D, numberOfPoints, INFINITY);
    fillIntArray(merged, numberOfPoints, 0);
    for(int k=0; k < numberOfPoints-1; k++) {
        currentMin = INFINITY;
        merged[x] = 1;
        for(int i=0; i < numberOfPoints; i++) {
            if(merged[i] != 1) {
                dist = condensedMatrixData[condensedIndex(numberOfPoints, x, i)];
                if(dist < D[i]) {
                    D[i] = dist;
                    DIndex[i] = x;
                }
                if(D[i] < currentMin) {
                    currentMin = D[i];
                    y = i;
                }
            }
        }
        persistenceDiagramData[k] = currentMin;
        persistenceIndicesData[k] = condensedIndex(numberOfPoints, DIndex[y], y);
        x = y;
    }
    // Free memory
    free(D);
    free(DIndex);
    free(merged);
    Py_DECREF(condensedMatrix);
    return PyTuple_Pack(2, persistenceDiagram, persistenceIndices);
}

static PyMethodDef ZeroPersistenceDiagramMethods[] = {
    {"zero_persistence_diagram_by_single_linkage_algorithm",
     zero_persistence_diagram_by_single_linkage_algorithm, METH_VARARGS,
     "Compute the zero persistence diagram of a point cloud using the single linkage algorithm."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef zero_persistence_diagram_module = {
    PyModuleDef_HEAD_INIT,
    "zero_persistence_diagram",
    "Compute the zero persistence diagram of a point cloud using the single linkage algorithm.",
    -1,
    ZeroPersistenceDiagramMethods
};

PyMODINIT_FUNC PyInit_zero_persistence_diagram(void) {
    import_array();
    return PyModule_Create(&zero_persistence_diagram_module);
}