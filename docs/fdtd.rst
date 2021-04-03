.. currentmodule:: fdtd

API Reference
===============

The following section outlines the API of fdtd.py.

Materials
---------

.. autoclass:: Material
    :members:


Objects
---------

Brick
~~~~~~~~~~~~~~~~~~

.. autoclass:: Brick

Sphere
~~~~~~~~~~~~~~~~~~

.. autoclass:: Sphere


Lumped Elements
------------------

Resistor
~~~~~~~~~~~~~~~~~~
.. autoclass:: Resistor

Capacitor
~~~~~~~~~~~~~~~~~~
.. autoclass:: Capacitor

Inductor
~~~~~~~~~~~~~~~~~~
.. autoclass:: Inductor


Sources
------------------

Voltage Source
~~~~~~~~~~~~~~~~~~
.. autoclass:: VoltageSource

Electric Field Source
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: EFieldSource

Impressed Electric Current Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ImpressedElectricCurrentSource

Impressed Magnetic Current Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ImpressedMagneticCurrentSource


Waveforms
------------------

Sine Waveform
~~~~~~~~~~~~~~~~~~
.. autoclass:: SineWaveform

Step Waveform
~~~~~~~~~~~~~~~~~~
.. autoclass:: StepWaveform

Gaussian Waveform
~~~~~~~~~~~~~~~~~~
.. autoclass:: GaussianWaveform

Detectors
------------------

Voltage Detector
~~~~~~~~~~~~~~~~~~
.. autoclass:: VoltageDetector

Magnetic Field Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: HFieldDetector

Electric Field Detector
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: EFieldDetector


Boundaries
------------------

Periodic Boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PeriodicBoundary

Periodic Bloch Boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: PeriodicBlochBoundary
