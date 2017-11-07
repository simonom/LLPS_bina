# vim: set et sw=4 sts=4 fileencoding=utf-8:
#
# Copyright (c) 2016 David Topping.
# All Rights Reserved.
# This file is part of umansysprop.
#
# umansysprop is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# umansysprop is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# umansysprop.  If not, see <http://www.gnu.org/licenses/>.

"""
CCN activation potential [Inorganic systems]
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')


from itertools import chain, product

from ..forms import (
    Form,
    SMILESDictField,
    FloatRangeField,
    FloatField,
    CoreAbundanceField,
    SizedependanceField,
    SelectField,
    InputRequired,
    Length,
    NumberRange,
    ZeroIonCharge,
    )
from ..results import Result, Table

from .. import boiling_points
from .. import vapour_pressures
from .. import CCN_activation
from .. import critical_properties
from .. import liquid_densities

class HandlerForm(Form):
    inorganic_ions = SMILESDictField(
        'Inorganic ions', entry_label='SMILES', data_label='moles',
        compounds=[
            ('[Na+]',             'Sodium cation'),
            ('[K+]',              'Potassium cation'),
            ('[NH4+]',            'Ammonium cation'),
            ('[Ca+2]',            'Calcium cation'),
            ('[Mg+2]',            'Magnesium cation'),
            ('[Cl-]',             'Chloride anion'),
            ('[O-][N+]([O-])=O',  'Nitrate anion'),
            ('[O-]S([O-])(=O)=O', 'Sulphate anion'),
            ],
        validators=[
            ZeroIonCharge(),
            ])
    interactions_method = SelectField(
        'Interaction model (please note calculating activity coefficients can significantly increase model simulation time)', default='nonideal', choices=[
            ('nonideal',  'Assume non-ideal interactions using AIOMFAC activity model'),
            ('ideal',     'Assume ideality'),
            ], validators=[InputRequired()])
    #size_dependance = SizedependanceField(
    #    'The diameter of the dry, spherical, particle with prescribed surface tension of the aqueous droplet'
    #    , validators=[InputRequired()])
    surface_tension=FloatField(
        'Surface tension at the point of activation (mN/m)', default=72.0, validators=[
        NumberRange(min=20.0, max=140.0)])
    dry_sizes = FloatRangeField(
        'Dry diameters (nm)', default=100, validators=[
            NumberRange(min=10.0, max=10000.0),
            Length(min=1, max=20, message='Sizes must have between 1 and 20 values'),
            ])
    temperatures = FloatRangeField(
        'Temperatures (K)', default=298.15, validators=[
            NumberRange(min=173.15, max=400.0),
            Length(min=1, max=20, message='Temperatures must have between 1 and 20 values'),
            ])


def handler(
        inorganic_ions, interactions_method, dry_sizes, surface_tension,
        temperatures):
    """
    Calculates the CCN activation potential of an inorganic mixture from a
    prescribed total condensed phase concentration, existing core and ambient
    conditions (all given *temperatures*). The *interactions_method* parameter
    is one of the strings:

    * 'nonideal'
    * 'ideal'

    Where 'nonideal' assumes non-ideal interactions using the AIOMFAC activity
    model [AIOMFAC]_, and 'ideal' assumes ideality.

    The *temperatures* are given as a sequence of floating point values (K).
    The *surface_tension* at the point of activation is given as a floating
    point value (mN/m), and *dry_sizes* is a sequence of floating point values
    (nm). The *inorganic_ions* are given as a sequence of SMILES strings.

    The result is a single table with the temperatures (K), dry diameter (nm),
    critical saturation ration of water vapour (%) and Kappa Kohler value
    (unitless) derived from the point of activation.


    .. [AIOMFAC]     Zuend, A., Marcolli, C., Luo, B. P., and Peter, T.: A
                     thermodynamic model of mixed organic-inorganic aerosols to
                     predict activity coefficients, Atmos. Chem. Phys., 8,
                     4559-4593, doi:10.5194/acp-8-4559-2008, 2008.
     """
    ideality = interactions_method #== 'ideal'
    #diameter, surface_tension =

    equilib_vp = {}
    totals = {}
    for temperature in temperatures:
        for dry_size in dry_sizes:
             sat_ratio, Kappa  = CCN_activation.growth_factor_model_inorg(
                   inorganic_ions, temperature, ideality, dry_size,surface_tension)
             #totals[(temperature, humidity), 'Water activity'] = water_activity
             totals[(temperature,dry_size), 'Critical ratio (%)'] = sat_ratio
             totals[(temperature,dry_size), 'Kappa'] = Kappa


    return Result(
        Table(
            'totals',
            title='Critical saturation ratio (%) and Kappa value (unitless) of aerosol particles',
            rows_title=('Temperature', 'Dry size'), rows_unit=('K', 'nm'), rows=product(temperatures, dry_sizes),
            cols_title='Result', cols=('Critical ratio (%)', 'Kappa'),
            data=totals,
            )
        )
