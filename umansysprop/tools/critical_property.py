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
Critical properties of organic compounds
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


class HandlerForm(Form):
    compounds = SMILESListField(
        'Compounds',
        compounds=[
            ('C(CC(=O)O)C(=O)O',                 'Succinic acid'),
            ('C(=O)(C(=O)O)O',                   'Oxalic acid'),
            ('O=C(O)CC(O)=O',                    'Malonic acid'),
            ('CCCCC/C=C/C/C=C/CC/C=C/CCCC(=O)O', 'Pinolenic acid'),
            ])
    properties_method = SelectField(
        'Critical properties', choices=[
            ('nannoolal',        'Nannoolal et al 2004'),
            ('joback_and_reid',  'Joback and Reid 1987'),
            ], validators=[InputRequired()])
    bp_method = SelectField(
        'Boiling point method', choices=[
            ('joback_and_reid',  'Joback and Reid 1987'),
            ('stein_and_brown',  'Stein and Brown 1994'),
            ('nannoolal',        'Nannoolal et al 2004'),
            ], validators=[InputRequired()])

def handler(compounds, properties_method, bp_method):
    """
    Calculates the critical properties for all specified *compounds*
    (given as a sequence of SMILES strings) The *properties_method* 
    parameter is one of the strings:

    * 'nannoolal'
    * 'joback_and_reid'

    Indicating whether to use the algorithms by Nannoolal et al [NANN2004]_ or
    Joback and Reid [JOBACK1987]_ when calculating the critical properties of
    each compound.

    The result is a single table with the
    compounds in the columns. Critical properties are given in..

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
    boiling_point = {
        'joback_and_reid': boiling_points.joback_and_reid,
        'stein_and_brown': boiling_points.stein_and_brown,
        'nannoolal':       boiling_points.nannoolal,
        }[bp_method]
    critical_property = {
        'nannoolal':          critical_properties.nannoolal,
        'joback_and_reid':    critical_properties.joback_and_reid,
        }[properties_method]
   

    data_temp = {}
    data_pressure = {}
    data_volume = {}
    data_compressibility = {}
    for c in compounds:
        b = boiling_point(c)
        critical_data=critical_property(c, b)
        data_temp[('1'),(str(c).strip())] = critical_data.temperature
        data_pressure[('1'),(str(c).strip())] = critical_data.pressure
        data_volume[('1'),(str(c).strip())] = critical_data.volume
        data_compressibility[('1'),(str(c).strip())] = critical_data.compressibility

    return Result(
        Table(
            'data_temp',
            title='Critical temperature (K)',rows=('1'),
            cols_title='Compound', cols=(str(c).strip() for c in compounds),
            data=data_temp,
            ),
        Table(
            'pressures',
            title='Critical pressure (bar)',rows=('1'),
            cols_title='Compound', cols=(str(c).strip() for c in compounds),
            data=data_pressure,
            ),
        Table(
            'volume',
            title='Critical volume (cm3/mol)',rows=('1'),
            cols_title='Compound', cols=(str(c).strip() for c in compounds),
            data=data_volume,
            ),
        Table(
            'compressibility',
            title='Critical compressib-ility',rows=('1'),
            cols_title='Compound', cols=(str(c).strip() for c in compounds),
            data=data_compressibility,
            )
        )








