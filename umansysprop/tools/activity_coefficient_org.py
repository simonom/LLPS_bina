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
Activity coefficients in liquids [Mixed Organic systems]
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
    CoreAbundanceField,
    SelectField,
    InputRequired,
    Length,
    NumberRange,
    ZeroIonCharge,
    )
from ..results import Result, Table

from .. import activity_coefficient_models

class HandlerForm(Form):
    organic_compounds = SMILESDictField(
        'Organic compounds', entry_label='SMILES', data_label='relative molar concentration',
        compounds=[
            ('C(CC(=O)O)C(=O)O',                 'Succinic acid'),
            ('C(=O)(C(=O)O)O',                   'Oxalic acid'),
            ('O=C(O)CC(O)=O',                    'Malonic acid'),
            ('O', 'Water'),
            ])
    interactions_method = SelectField(
        'Interaction model', default='UNIFAC', choices=[
            ('AIOMFAC',  'Assume non-ideal interactions using AIOMFAC activity model'),
            ], validators=[InputRequired()])
    temperatures = FloatRangeField(
        'Temperatures (K)', default=298.15, validators=[
            NumberRange(min=173.15, max=400.0),
            Length(min=1, max=100, message='Temperatures must have between 1 and 100 values'),
            ])

def handler(organic_compounds, interactions_method, temperatures):
    """
    Calculates the activity coefficients for all components, including
    dissociated ions (all given *temperatures*, a sequence of floating point
    values in Kelvin).

    The *interactions_method* parameter is one of the strings:

    Activity coefficients are predicted assuming a homogeneous bulk representation,
    allowing all compounds to interact according to the technique applied. No
    partitioning between the liquid and another phase is accounted for. 

    * 'AIOMFAC' - assume non-ideal interactions using the [AIOMFAC]_ model.

    This is currently restricted only to AIOMFAC variants. This includes the
    UNIFAC component of the larger AIOMFAC model, dealing with only short-range
    interactions. UNIFAC can calculate the activity coefficients of inorganic
    ions but the reference state would need to be corrected.  Using AIOMFAC,
    this is done automatically.

    The result is a table with Temperature (K) provided in the rows and
    compound activity coefficient (on the mole fraction scale), with associated
    SMILES string, given in each column.

    .. [AIOMFAC]  Zuend, A., Marcolli, C., Luo, B. P., and Peter, T.: A
                  thermodynamic model of mixed organic-inorganic aerosols to
                  predict activity coefficients, Atmos. Chem. Phys., 8,
                  4559-4593, doi:10.5194/acp-8-4559-2008, 2008.
     """

    activity_coefficient = {
        'AIOMFAC':  activity_coefficient_models.calculate_activities_org,
        }[interactions_method]

    coefficients = {}
    for temperature in temperatures:
        activity_result, m = activity_coefficient(organic_compounds, temperature)
        for c in m:
            coefficients[(temperature), str(c.compound).strip()] = c.activity_coefficient

    return Result(
        Table(
            'coefficients',
            title='Activity coefficients (unitless, on the mole fraction scale) in the one liquid phase',
            rows_title=('Temperature'), rows_unit=('K'), rows=temperatures,
            cols_title='Compound', cols=[str(c.compound).strip() for c in m],
            data=coefficients,
            )
        )
