#include <Python.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <list>
#include <vector>
#include <cassert>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <metis.h>
#include <numpy/arrayobject.h>

static char doc_splitNodeMesh[] =
    "Usage : metis.splitNodeMesh( nbDoms, (begRows, indCols) )"
    "where nbDoms is the number of subdomains"
    "and (begRows,indCols) are the graph of the matrix."
    "return set of nodes per domains\n";
static PyObject* py_splitNodeMesh( PyObject* self, PyObject* args ) {
    PyArrayObject *pyBegRows, *pyIndCols;
    int            nbSubDomains;

    if ( !PyArg_ParseTuple( args, "i(O!O!)", &nbSubDomains, &PyArray_Type, &pyBegRows, &PyArray_Type, &pyIndCols ) )
        return NULL;
    const long* begRows = (const long*)PyArray_DATA( pyBegRows );
    const long* indCols = (const long*)PyArray_DATA( pyIndCols );
    assert(begRows != nullptr);
    assert(indCols != nullptr);

    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions( options );
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_PTYPE]     = METIS_PTYPE_RB;

    idx_t  nbVerts = idx_t( PyArray_DIM( pyBegRows, 0 ) ) - 1;
    idx_t  ncon    = 1;
    std::vector<idx_t> xadj(nbVerts + 1);
    std::copy( begRows, begRows + nbVerts + 1, xadj.begin() );
    std::vector<idx_t> adjncy(xadj[nbVerts]);
    std::copy( indCols, indCols + begRows[nbVerts], adjncy.begin() );
    idx_t  nbDoms = nbSubDomains;
    idx_t  objval;
    std::vector<idx_t> part(nbVerts);
    /*int ok =*/
    /*METIS_PartGraphKway( &nbVerts, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL, 
                         &nbDoms, NULL, NULL, options, &objval, part.data() );*/
    METIS_PartGraphRecursive(&nbVerts, &ncon, xadj.data(), adjncy.data(), NULL, NULL, NULL, &nbDoms, 
                             NULL, NULL, options, &objval, part.data() );
    /* On compte maintenant le nombre de noeuds appartenant à chaque domaine : */
    std::vector<long> nbNodesPerDomains(nbSubDomains,0);
    for ( int      i     = 0; i < nbVerts; ++i ) { nbNodesPerDomains[part[i]] += 1; }
    PyObject*      pyLst = PyList_New( nbSubDomains );
    PyArrayObject* indVertSubDomains;
    std::vector<long*> ptIndVertices(nbSubDomains);
    for ( long i = 0; i < nbSubDomains; ++i ) {
        npy_intp nbVerts  = nbNodesPerDomains[i];
        indVertSubDomains = (PyArrayObject*)PyArray_SimpleNew( 1, &nbVerts, NPY_LONG );
        PyList_SetItem( pyLst, i, (PyObject*)indVertSubDomains );
        ptIndVertices[i] = (long*)PyArray_DATA( indVertSubDomains );
    }
    std::fill_n( nbNodesPerDomains.begin(), nbSubDomains, 0 );
    for ( long i = 0; i < nbVerts; ++i ) {
        ptIndVertices[part[i]][nbNodesPerDomains[part[i]]] = i;
        nbNodesPerDomains[part[i]] += 1;
    }
    return Py_BuildValue( "N", pyLst );
}
// ------------------------------------------------------------------------
static char doc_splitEltMesh[] =
    "Usage : metis.splitEltMesh( nbDoms, nbVerts, elt2verts )"
    "where nbDoms is the number of subdomains, "
    "nbVerts the number of vertices in the mesh and "
    "elt2verts is the element to vertices connexion."
    "return a set of elements per domains\n";

static PyObject* py_splitEltMesh( PyObject* self, PyObject* args ) {
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions( options );
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_PTYPE]     = METIS_PTYPE_RB;
    options[METIS_OPTION_RTYPE]     = METIS_RTYPE_GREEDY;
    int            nbSubDoms, nbVerts;
    PyArrayObject* py_elt2verts;
    if ( !PyArg_ParseTuple( args, "iiO!", &nbSubDoms, &nbVerts, &PyArray_Type, &py_elt2verts ) ) return NULL;
    const long* elt2verts = (const long*)PyArray_DATA( py_elt2verts );
    idx_t       nn        = nbVerts;
    idx_t       ne        = idx_t( PyArray_DIM( py_elt2verts, 0 ) );

    std::vector<idx_t> eptr(ne+1);
    eptr[0]     = 0;
    for ( long i = 1; i <= ne; ++i ) eptr[i] = eptr[i - 1] + 3;
    std::vector<idx_t> eind(3*ne);
    std::copy( elt2verts, elt2verts + 3 * ne, eind.begin() );
    idx_t  ncommon = 2;
    idx_t  nparts  = nbSubDoms;
    idx_t  objval = 0;
    std::vector<idx_t> epart(ne);
    std::vector<idx_t> npart(nn);
    /*int ok =*/
    METIS_PartMeshDual( &ne, &nn, eptr.data(), eind.data(), NULL, NULL, &ncommon, &nparts, 
                       NULL, options, &objval, epart.data(), npart.data() );

    /* On compte le nombre d'éléments par domaine : */
    std::vector<long> nbEltsPerDoms(nbSubDoms,0);
    for ( long     i     = 0; i < ne; ++i ) { nbEltsPerDoms[epart[i]] += 1; }
    PyObject*      pyLst = PyList_New( nbSubDoms );
    PyArrayObject* indEltSubDoms;
    long**         ptIndElts = new long*[nbSubDoms];
    for ( long i = 0; i < nbSubDoms; ++i ) {
        npy_intp nbElts = nbEltsPerDoms[i];
        indEltSubDoms   = (PyArrayObject*)PyArray_SimpleNew( 1, &nbElts, NPY_LONG );
        PyList_SetItem( pyLst, i, (PyObject*)indEltSubDoms );
        ptIndElts[i] = (long*)PyArray_DATA( indEltSubDoms );
    }
    std::fill_n( nbEltsPerDoms.begin(), nbSubDoms, 0 );
    for ( long i = 0; i < ne; ++i ) {
        ptIndElts[epart[i]][nbEltsPerDoms[epart[i]]] = i;
        nbEltsPerDoms[epart[i]] += 1;
    }

    return Py_BuildValue( "N", pyLst );
}
// ========================================================================
static PyMethodDef Py_Methods[] = {
    {"splitNodeMesh", py_splitNodeMesh, METH_VARARGS, doc_splitNodeMesh},
    {"splitEltMesh", py_splitEltMesh, METH_VARARGS, doc_splitEltMesh},
    {NULL, NULL} /* Guards */
};
// ========================================================================
static char    splitter_doc[] = "Partitionner of mesh using metis";
PyMODINIT_FUNC initsplitter( ) {
    PyObject* m;
    m = Py_InitModule4( "splitter", Py_Methods, splitter_doc, (PyObject*)NULL, PYTHON_API_VERSION );
    if ( m == NULL ) return;
    /*  important : initialize numpy to use in subroutines !!!! */
    import_array( );
}
