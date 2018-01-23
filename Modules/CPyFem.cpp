# include <Python.h>
# include <math.h>
# include <list>
# include <algorithm>
# include <iostream>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# include <numpy/arrayobject.h>

static char compSkelSparseMatrix_doc[] = 
  "Usage : compute_skeleton_SparseMatrix( elt2dofs, (pt_dof2elts, dof2elts) )\n"
  "Calcul le graphe de la matrice creuse issue des éléments finis.\n"
  "elt2dofs    : Tableau donnant pour chaque élément les degrés de liberté associés. \n"
  "pt_dof2elts : pointeurs dans le tableau dof2elts donnant pour chaque degré de     \n"
  "              liberté les éléments associés.\n"
  "dof2elts    : tableau donnant pour chaque degré de liberté les éléments associés.\n"
  "\n"
  "Attention : Les tableaux doivent contenir des entiers longs !!!\n"
  "---------\n";
static PyObject* py_compSkelSparseMatrix( PyObject* self, PyObject* args )
{
  PyArrayObject *PyElt2Dofs, *PyBegDof2Elts, *PyDof2Elts;

  if ( !PyArg_ParseTuple( args, "O!(O!O!)",
			  &PyArray_Type, &PyElt2Dofs,
			  &PyArray_Type, &PyBegDof2Elts,
			  &PyArray_Type, &PyDof2Elts ) ) return NULL;
  const long* elt2dofs    = (const long*)PyArray_DATA(PyElt2Dofs);
  const long* begDof2Elts = (const long*)PyArray_DATA(PyBegDof2Elts);
  const long* dof2elts    = (const long*)PyArray_DATA(PyDof2Elts);

  long nbDofs = long(PyArray_DIM(PyBegDof2Elts,0))-1;

  npy_intp nbRowsP1, nbNZ;
  nbRowsP1 = npy_intp(nbDofs) + 1;
  PyArrayObject* pyBegRows = (PyArrayObject*) PyArray_SimpleNew(1,&nbRowsP1,NPY_LONG);
  long* begRows = (long*)PyArray_DATA(pyBegRows);

  begRows[0] = 0;
  for ( int idof = 0; idof < nbDofs; ++idof ) {
    std::list<long> lst_neighbours;
    for ( long ptIElt = begDof2Elts[idof]; ptIElt < begDof2Elts[idof+1]; ++ptIElt ) {
      long iElt = dof2elts[ptIElt];
      for ( long ptDof = 3*iElt; ptDof < 3*(iElt+1); ptDof++ ) {
	lst_neighbours.push_back(elt2dofs[ptDof]);
      }
    }
    lst_neighbours.sort();
    lst_neighbours.unique();
    begRows[idof+1] = begRows[idof] + lst_neighbours.size();
  }
  
  nbNZ = begRows[nbDofs];
  PyArrayObject* pyIndCols = (PyArrayObject*) PyArray_SimpleNew(1,&nbNZ,NPY_LONG);
  long* indCols = (long*)PyArray_DATA(pyIndCols);

  for ( long idof = 0; idof < nbDofs; ++idof ) {
    std::list<long> lst_neighbours;
    for ( long ptIElt = begDof2Elts[idof]; ptIElt < begDof2Elts[idof+1]; ++ptIElt ) {
      long iElt = dof2elts[ptIElt];
      for ( long ptDof = 3*iElt; ptDof < 3*(iElt+1); ptDof++ ) {
	lst_neighbours.push_back(elt2dofs[ptDof]);
      }
    }
    lst_neighbours.sort();
    lst_neighbours.unique();
    long ind = 0;
    for ( std::list<long>::iterator itL = lst_neighbours.begin(); itL != lst_neighbours.end(); ++itL ) {
      indCols[begRows[idof]+ind] = (*itL);
      ind += 1;
    }
  }

  return Py_BuildValue("NN", pyBegRows, pyIndCols);
}
// ------------------------------------------------------------------------
static char addElementaryMatrixToCSCMatrix_doc[] = 
"Usage : add_elemMat_csrMatrix( (begCols, indRows, coefs), (indRows, indCols, elemMat) )"
"Rajoute la matrice élémentaire définie par le tuple (indRows, indCols,elemMat)"
"à la matrice creuse stockée CSC définie par le tuple (begCols, indRows, coefs).";
static PyObject*
py_addelementmatrix_cscmatrix( PyObject* self, PyObject* args )
{
  // Tableaux pour la matrice creuse :
  PyArrayObject *pysm_BegCols, *pysm_IndRows, *pysm_Coefs;
  // Tableaux pour la matrice élémentaire :
  PyArrayObject *pyem_IndRows, *pyem_IndCols, *pyem_Coefs;
  //
  if ( !PyArg_ParseTuple( args, "(O!O!O!)(O!O!O!)",
			  &PyArray_Type, &pysm_BegCols,
			  &PyArray_Type, &pysm_IndRows,
			  &PyArray_Type, &pysm_Coefs,
			  &PyArray_Type, &pyem_IndRows,
			  &PyArray_Type, &pyem_IndCols,
			  &PyArray_Type, &pyem_Coefs ) ) return NULL;
  const long* sm_begcols = (const long*)PyArray_DATA(pysm_BegCols);
  const long* sm_indrows = (const long*)PyArray_DATA(pysm_IndRows);
  double    * sm_coefs   = (   double*)PyArray_DATA(pysm_Coefs  );
  
  long nRowsMatElem = long(PyArray_DIM(pyem_IndRows,0));
  long nColsMatElem = long(PyArray_DIM(pyem_IndCols,0));
  const long* em_indrows = (const long*)PyArray_DATA(pyem_IndRows);
  const long* em_indcols = (const long*)PyArray_DATA(pyem_IndCols);

  for (long iRow = 0; iRow < nRowsMatElem; ++iRow ) {
    long indRow = em_indrows[iRow];
    for ( long jCol = 0; jCol < nColsMatElem; ++jCol ) {
      long indCol = em_indcols[jCol];
      for ( long ptRow = sm_begcols[indCol]; ptRow<sm_begcols[indCol+1];++ptRow ) {
	if ( sm_indrows[ptRow] == indRow ) {
	  sm_coefs[ptRow] += *(double*)PyArray_GETPTR2(pyem_Coefs,iRow,jCol);
	  break;
	}
      }
    }
  }
  Py_INCREF(Py_None);
  return Py_None;
}
// ========================================================================
static PyMethodDef Py_Methods[] =
  {
    {"comp_skel_csc_mat", py_compSkelSparseMatrix, METH_VARARGS, compSkelSparseMatrix_doc},
    {"add_elt_mat_to_csc_mat", py_addelementmatrix_cscmatrix, 
     METH_VARARGS, addElementaryMatrixToCSCMatrix_doc},
    {NULL, NULL} /* Guards */
  };
// ========================================================================
static char fem_doc[] = "Finite Element Method";
PyMODINIT_FUNC initfem()
{
  PyObject* m; 
  m = Py_InitModule4("fem",Py_Methods, fem_doc,
		     (PyObject*)NULL,PYTHON_API_VERSION);
  if (m==NULL) return;
  /*  important : initialize numpy to use in subroutines !!!! */
  import_array();
}
