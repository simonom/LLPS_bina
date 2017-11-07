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
Sub-cooled density of organic compounds
"""

from __future__ import (
    unicode_literals,
    absolute_import,
    print_function,
    division,
    )
str = type('')


from ..forms import (
    Form,
    FloatRangeField,
    SMILESListField,
    SelectField,
    InputRequired,
    Length,
    NumberRange,
    )
from ..results import Result, Table

from .. import boiling_points
from .. import critical_properties
from .. import liquid_densities


class HandlerForm(Form):
    compounds = SMILESListField(
        'Compounds',
        compounds=[
            ('C(CC(=O)O)C(=O)O',                 'Succinic acid'),
            ('C(=O)(C(=O)O)O',                   'Oxalic acid'),
            ('O=C(O)CC(O)=O',                    'Malonic acid'),
            ('CCCCC/C=C/C/C=C/CC/C=C/CCCC(=O)O', 'Pinolenic acid'),
            ])
    temperatures = FloatRangeField(
        'Temperatures (K)', default=298.15, validators=[
            NumberRange(min=173.15, max=400.0, message='Temperatures must be between 173.15K and 400.0K'),
            Length(min=1, max=100, message='Temperatures must have between 1 and 100 values'),
            ])
    density_method = SelectField(
        'Sub-cooled density predictive technique', choices=[
            ('girolami',      'Girolami 1994'),
            ('schroeder',     'Schroeder (Poling et al 2001)'),
            ('le_bas',        'Le Bas 1915'),
            ('tyn_and_calus', 'Tyn and Calus 1975'),
            ], validators=[InputRequired()])
    properties_method = SelectField(
        'Critical properties', choices=[
            ('nannoolal',        'Nannoolal et al 2004'),
            ('joback_and_reid',  'Joback and Reid 1987'),
            ], validators=[InputRequired()])


def handler(compounds, temperatures, density_method, properties_method):
    """
    Calculates the sub-cooled density for all specified *compounds* (given as a
    sequence of SMILES strings) at all given *temperatures* (a sequence of
    floating point values giving temperatues in degrees Kelvin).  The
    *density_method* parameter is one of the strings:

    * 'girolami'
    * 'schroeder'
    * 'le_bas'
    * 'tyn_and_calus'

    Indicating whether to use the algorithm by Girolami [GIRO1994]_, Schroeder,
    Le Bas [LEBAS1915]_, or Tyn and Calus [TYN1975]_ when calculating the
    density. Finally, the *properties_method* parameter is one of the strings:

    * 'nannoolal'
    * 'joback_and_reid'

    Indicating whether to use the algorithms by Nannoolal et al [NANN2004]_ or
    Joback and Reid [JOBACK1987]_ when calculating the critical properties of
    each compound.

    The result is a single table with the temperatures in the rows, and the
    compounds in the columns. Liquid densities are given as grams per cubic
    centimetre (g/cc).

    .. [GIRO1994]    Girolami,G.S., Journal of Chemical Education, 1994, 71,
                     962–964.
    .. [LEBAS1915]   Le Bas, G. The Molecular Volume of Liquid Chemical
                     Compounds; Longmans, Green: New York, NY, USA, 1915.
    .. [TYN1975]     Tyn, M.T. and W. F. Calus, Processing 1975, 21, 16.
    .. [NANN2004]    Nannoolal, Y., Rarey, J., Ramjugernath, D., and
                     Cordes, W.: Estimation of pure component properties
                     Part 1, Estimation of the normal boiling point of
                     non-electrolyte organic compounds via group
                     contributions and group interactions, Fluid Phase
                     Equilibr., 226, 45–63, 2004
    .. [JOBACK1987]  Joback, K. G. and Reid, R. C.: Estimation of
                     pure-component properties from group-contributions,
                     Chem. Eng. Commun., 57, 233– 243, 1987.
    """
    critical_property = {
        'nannoolal':          critical_properties.nannoolal,
        'joback_and_reid':    critical_properties.joback_and_reid,
        }[properties_method]
    liquid_density = {
        'girolami':      lambda c, t, p: liquid_densities.girolami(c),
        'schroeder':     liquid_densities.schroeder,
        'le_bas':        liquid_densities.le_bas,
        'tyn_and_calus': liquid_densities.tyn_and_calus,
        }[density_method]

    data = {}
    for c in compounds:
        b = boiling_points.nannoolal(c)
        for t in temperatures:
            data[(t, str(c).strip())] = liquid_density(c, t, critical_property(c, b))

    return Result(
        Table(
            'densities',
            title='Sub-cooled liquid density in grams per cubic centimetre (g/cc)',
            rows_title='Temperature (K)', rows=temperatures,
            cols_title='Compound', cols=(str(c).strip() for c in compounds),
            data=data,
            )
        )

