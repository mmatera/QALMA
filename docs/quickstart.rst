Quickstart
==========

This quickstart guide shows how to get started with QALMA: building quantum systems, exploring their structure, manipulating operators, and simulating time evolution.


Installing
----------

The most straightforward way to install the library is using PIP, from the
GitHub repo:

.. code-block:: bash

    pip install git+https://github.com/QILPCM-IFLP-CONICET/QALMA/

This command is going to download and install the package and its dependencies. Notice that if your Python installation is in your root filesystem, the command must be run using root privileges (for example, using ``sudo``).

A better option is to install it in a virtual environment. To install it in a virtual environment,


1.  **Create the virtual environment**

    In a terminal, execute the following command to create a virtual environment called ``qalma_env``:

    .. code-block:: bash

        python -m venv qalma_env

2.  **Activate the virtual environment**

    Once created, activate the virtual environment:

    *   In macOS and Linux systems:

        .. code-block:: bash

            source qalma_env/bin/activate

    *   In Windows systems, using the *Command Prompt*:

        .. code-block:: bash

            .\qalma_env\Scripts\activate.bat

    *   In Windows systems using *PowerShell*:

        .. code-block:: bash

            .\qalma_env\Scripts\Activate.ps1

3.  **Install QALMA**

    Once the virtual environment is active, we can proceed to install
    the QALMA library directly from its GitHub repo:

    .. code-block:: bash

        pip install git+https://github.com/QILPCM-IFLP-CONICET/QALMA/


4. **Verifying your installation**
   To verify that the installation was successful, open a Python terminal
   and run
     
   .. code-block:: python

      from qalma import build_system
      system = build_system()
      print(system)

   Output::

     graph:Graph chain lattice. Vertices: 				 
        1[0] of type {'type': '0', 'coords': array([0.86927373])}	 
        1[1] of type {'type': '0', 'coords': array([1.86927373])}	 
        1[2] of type {'type': '0', 'coords': array([2.86927373])}	 
        1[3] of type {'type': '0', 'coords': array([3.86927373])}	 
     Edges								 
      type 0:							 
         1[0]-1[1]							 
         1[1]-1[2]							 
         1[2]-1[3]							 
         1[3]-1[0]							 
     sites:dict_keys(['1[0]', '1[1]', '1[2]', '1[3]'])		 
     dimensions:{'1[0]': 2, '1[1]': 2, '1[2]': 2, '1[3]': 2}


Once the installation is complete, QALMA and its dependencies will be available inside the virtual environment ``qalma_env``.




Importing Required Libraries
----------------------------

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np
    from qalma import build_system
    # Optional: list_models_in_alps_xml, list_geometries_in_alps_xml, graph_from_alps_xml

Building a Simple Quantum System
--------------------------------

To create a default system (a spin-1/2 chain with four sites), use:

.. code-block:: python

    system = build_system()

The system is described by a ``SystemDescriptor`` object, which contains information about the model, its geometry, and parameters. By default, ``build_system()`` creates a periodic spin-1/2 chain with four sites.


Visualizing the Lattice
-----------------------

You can access and draw the underlying graph:

.. code-block:: python

    system.spec["graph"].draw(plt)
    plt.show()


.. image:: figures/quick_start/fig_1.png
   :alt: Output
   :align: center
   :width: 400px

    
Exploring Sites and Local Properties
------------------------------------

Each site is described in the ``system.sites`` dictionary:

.. code-block:: python

    print(system.sites.keys())  # Lists all site names

    # Explore the first site's properties:
    site = system.sites['1[0]']
    print("The dimension of the first site is ", site["dimension"])
    print("Quantum numbers:", site["qn"])
    print("Operators:", tuple(site["operators"]))

Output::

    dict_keys(['1[0]', '1[1]', '1[2]', '1[3]'])
    the dimension of the first site is  2
    Quantum numbers: {'S': {'min': 0.5, 'max': 0.5, 'fermionic': False, 'operator': 's'}, 'Sz': {'min': '-S', 'max': 'S', 'fermionic': False, 'operator': 'Sz'}}
    Operators: ('identity', 'Splus', 'Sminus', 'Sz', 's', 'Sx', 'Sy')
    

Working with Operators
----------------------

Through a ``SystemDescriptor`` object, its global operators (such as the Hamiltonian and magnetization) are now available through the ``global_operator`` method:

.. code-block:: python

    H = system.global_operator("Hamiltonian")
    print(H)

Output::

  (											       
  qutip interface operator over sites {'1[0]': 0, '1[1]': 1} for 1 x  			       
  Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=CSR, isherm=True       
  Qobj data =										       
  [[ 0.25  0.    0.    0.  ]								       
   [ 0.   -0.25  0.5   0.  ]								       
   [ 0.    0.5  -0.25  0.  ]								       
   [ 0.    0.    0.    0.25]]								       
    +qutip interface operator over sites {'1[1]': 0, '1[2]': 1} for 1 x  			       
  Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=CSR, isherm=True       
  Qobj data =										       
  [[ 0.25  0.    0.    0.  ]								       
   [ 0.   -0.25  0.5   0.  ]								       
   [ 0.    0.5  -0.25  0.  ]								       
   [ 0.    0.    0.    0.25]]								       
    +qutip interface operator over sites {'1[2]': 0, '1[3]': 1} for 1 x  			       
  Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=CSR, isherm=True       
  Qobj data =										       
  [[ 0.25  0.    0.    0.  ]								       
   [ 0.   -0.25  0.5   0.  ]								       
   [ 0.    0.5  -0.25  0.  ]								       
   [ 0.    0.    0.    0.25]]								       
    +qutip interface operator over sites {'1[0]': 0, '1[3]': 1} for 1 x  			       
  Quantum object: dims=[[2, 2], [2, 2]], shape=(4, 4), type='oper', dtype=CSR, isherm=True       
  Qobj data =										       
  [[ 0.25  0.    0.    0.  ]								       
   [ 0.   -0.25  0.5   0.  ]								       
   [ 0.    0.5  -0.25  0.  ]								       
   [ 0.    0.    0.    0.25]]								       
  )  											       

o, en una interface de Jupyter Notebook



  
.. math::

  \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[0],1[1]}
  + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[1],1[2]}
  + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[2],1[3]}
  + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[0],1[3]}


.. code-block:: python

    Sz = system.global_operator("Sz")
    print(Sz)

.. math::

   \left(\begin{array}{cc}-0.500 & 0\\0 & 0.500\end{array}\right)_{1[0]}
   + \left(\begin{array}{cc}-0.500 & 0\\0 & 0.500\end{array}\right)_{1[1]}
     + \left(\begin{array}{cc}-0.500 & 0\\0 & 0.500\end{array}\right)_{1[2]}
       + \left(\begin{array}{cc}-0.500 & 0\\0 & 0.500\end{array}\right)_{1[3]}


Or a local operator acting on a specific site:
   
.. code-block:: python

    Sx1 = system.site_operator("Sx@1[0]")
    print(Sx1)

.. math::

   \left(\begin{array}{cc}0 & 0.500\\0.500 & 0\end{array}\right)_{1[0]}
   
You can view the list of predefined global operators:

.. code-block:: python

    print(tuple(system.operators["global_operators"]))

Output::

   ('Sz', 'loop_term', 'spin_exchange_energy', 'Hamiltonian')


Operators can be combined algebraically:

.. code-block:: python

    Hzeeman = -2 * Sz
    Htotal = (Hzeeman + H).simplify()
    print(Htotal)


.. math::

   \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[0],1[1]}
   + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[1],1[2]}
   + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[2],1[3]}
   + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[0],1[3]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[0]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[1]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[2]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[3]}
    
Analyzing Operators
-------------------

You can compute eigenvalues, exponentiate, or take the trace of operators:

.. code-block:: python

    print("Energies:", Htotal.eigenenergies())      # Spectrum

Output::

   Energies: array([-3.00000000e+00, -3.00000000e+00, -2.00000000e+00, -2.00000000e+00,
       -2.00000000e+00, -1.00000000e+00, -1.00000000e+00, -1.22220204e-16,
        5.77179330e-18,  5.66309413e-16,  1.00000000e+00,  1.00000000e+00,
        2.00000000e+00,  2.00000000e+00,  3.00000000e+00,  5.00000000e+00])

.. code-block:: python

    print("Exp(H)=\n",Htotal.expm())               # Exponential


Output::

   Exp(H)=

.. math::

   \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[0],1[1]} +
   \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[1],1[2]}
   + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[2],1[3]}
   + \left(\begin{array}{cccc}0.250 & 0 & 0 & 0\\0 & -0.250 & 0.500 & 0\\0 & 0.500 & -0.250 & 0\\0 & 0 & 0 & 0.250\end{array}\right)_{1[0],1[3]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[0]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[1]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[2]}
   + \left(\begin{array}{cc}1 & 0\\0 & -1\end{array}\right)_{1[3]}


.. code-block:: python

   print("The partition function is ", (-Htotal).expm().tr(),"~", sum([np.exp(-en) for en in Htotal.eigenenergies()]))
   
Output::

   The partition function is  71.83776026426847 ~ 71.83776026426845
    

    
Visualizing Operator Support
----------------------------

To see which sites an operator acts on, use:

.. code-block:: python

    from qalma.utils import draw_operator
    fig, ax = plt.subplots()
    draw_operator(Htotal, ax)
    Htotal.system.spec["graph"].draw(ax)
    plt.show()


.. image:: figures/quick_start/fig_2.png
   :alt: Output
   :align: center
   :width: 400px

QuTiP Integration and Time Evolution
------------------------------------

Operators can be converted to QuTiP objects and used in QuTiP solvers:

.. code-block:: python

    import qutip

    sx01 = system.site_operator("Sx@1[0]") + system.site_operator("Sx@1[1]")
    ham = -sx01   # effective "local" hamiltonian
    rho0 = (-ham).expm()   # Gibbs-like state
    rho0 = rho0 / rho0.tr()
    ts = np.linspace(0, 10, 100)

    result = qutip.mesolve(
        H=Hzeeman.to_qutip(),
        rho0=rho0.to_qutip(),
        tlist=ts,
        e_ops=(sx01.to_qutip(),)
    )
    plt.plot(ts, result.expect[0], label="$H_{Zeeman}$")

    result = qutip.mesolve(
        H=H.to_qutip(),
        rho0=rho0.to_qutip(),
        tlist=ts,
        e_ops=(sx01.to_qutip(),)
    )
    plt.plot(ts, result.expect[0], label="$H_{exc}$")

    result = qutip.mesolve(
        H=Htotal.to_qutip(),
        rho0=rho0.to_qutip(),
        tlist=ts,
        e_ops=(sx01.to_qutip(),)
    )
    plt.plot(ts, result.expect[0], label="$H_{total}$")

    plt.legend()
    plt.xlabel("t")
    plt.ylabel(r"$\langle sx_1+sx_2\rangle$")
    plt.show()


.. image:: figures/quick_start/fig_3.png
   :alt: Output
   :align: center
   :width: 400px


Larger Systems
--------------

Larger quantum systems can be defined and manipulated in the same way, provided that operations requiring explicit matrix representations (such as full diagonalization) are avoided.

.. code-block:: python

    large_system = build_system(100)  # Adjust parameters for larger systems as needed
    sz = large_system.global_operator("Sz")
    H = large_system.global_operator("Hamiltonian") + sz

    sx0_loc=large_system.site_operator("Sx@1[0]")
    comm = (H * sx0_loc - sx0_loc * H).simplify()
    comm


.. math::

   \left(\begin{array}{cccc}0 & -0.250 & 0.250 & 0\\0.250 & 0 & 0 & -0.250\\-0.250 & 0 & 0 & 0.250\\0 & 0.250 & -0.250 & 0\end{array}\right)_{1[0],1[1]}
   + \left(\begin{array}{cccc}0 & -0.250 & 0.250 & 0\\0.250 & 0 & 0 & -0.250\\-0.250 & 0 & 0 & 0.250\\0 & 0.250 & -0.250 & 0\end{array}\right)_{1[0],1[99]}
   + \left(\begin{array}{cc}0 & -0.500\\0.500 & 0\end{array}\right)_{1[0]}
