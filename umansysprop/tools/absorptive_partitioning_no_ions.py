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
Equilibrium absorptive partitioning calculations [no inorganic core]
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

from .. import boiling_points
from .. import vapour_pressures
from .. import partition_models


class HandlerForm(Form):
    organic_compounds = SMILESDictField(
        'Organic compounds', entry_label='SMILES', data_label='molecules/cc',
        compounds=[
            ('C(CC(=O)O)C(=O)O',                 'Succinic acid'),
            ('C(=O)(C(=O)O)O',                   'Oxalic acid'),
            ('O=C(O)CC(O)=O',                    'Malonic acid'),
            ('CCCCC/C=C/C/C=C/CC/C=C/CCCC(=O)O', 'Pinolenic acid'),
            ])
    interactions_method = SelectField(
        'Interaction model', default='ideal', choices=[
            ('nonideal',  'Assume non-ideal interactions using AIOMFAC activity model'),
            ('ideal',     'Assume ideality'),
            ], validators=[InputRequired()])
    vp_method = SelectField(
        'Vapour pressure method', choices=[
            ('nannoolal',   'Nannoolal et al 2008'),
            ('myrdal',      'Myrdal and Yalkowsky 1997'),
            ('evaporation', 'Evaporation - Compernolle et al 2011'),
            ], validators=[InputRequired()])
    bp_method = SelectField(
        'Boiling point method', default='nannoolal', choices=[
            ('joback',    'Joback and Reid 1987'),
            ('stein',     'Stein and Brown 1994'),
            ('nannoolal', 'Nannoolal et al 2004'),
            ], validators=[InputRequired()])
    soluble_core = CoreAbundanceField(
        'Soluble, involatile, inert core abundance', soluble=True
        )
    insoluble_core = CoreAbundanceField(
        'Insoluble, involatile, inert core abundance', soluble=False
        )
    temperatures = FloatRangeField(
        'Temperatures (K)', default=298.15, validators=[
            NumberRange(min=173.15, max=400.0),
            Length(min=1, max=100, message='Temperatures must have between 1 and 100 values'),
            ])
    humidities = FloatRangeField(
        'Humidities (%)', default=50.0, validators=[
            NumberRange(min=0.0, max=100.0),
            Length(min=1, max=100, message='Relative humidities must have between 1 and 100 values'),
            ])


def handler(organic_compounds, interactions_method, vp_method,
        bp_method, soluble_core, insoluble_core, temperatures, humidities):
    """
    Calculates the condensed phase abundance of organic *compounds* from a
    prescribed total gaseous concentration, an existing core and ambient
    conditions (all given *temperatures* and relative *humidities*). The
    *vp_method* parameter is one of the strings:

    * 'nannoolal'
    * 'myrdal_and_yalkowsky'
    * 'evaporation'

    Indicating whether to use the methods of Nanoolal et al [NANN2008]_, Myrdal
    and Yalkowsky [MYRDAL1997]_, or evaporation [COMP2011]_ when calculating
    the vapour pressure. The *bp_method* parameter is one of the strings:

    * 'nannoolal'
    * 'joback_and_reid'
    * 'stein_and_brown'

    Indicating whether to use the methods of Nannoolal et al [NANN2004]_,
    Joback and Reid [JOBACK1987]_, or Stein and Brown [STEIN1994]_ when
    calculating the boiling point for each compound. The *interactions_method*
    parameter is one of the strings:

    * 'nonideal'
    * 'ideal'

    Where 'nonideal' assumes non-ideal interactions using the AIOMFAC activity
    model [AIOMFAC]_, and 'ideal' assumes ideality.

    The result is a set of several tables:

    .. [NANN2004]    Nannoolal, Y., Rarey, J., Ramjugernath, D., and
                     Cordes, W.: Estimation of pure component properties
                     Part 1, Estimation of the normal boiling point of
                     non-electrolyte organic compounds via group
                     contributions and group interactions, Fluid Phase
                     Equilibr., 226, 45–63, 2004
    .. [NANN2008]    Nannoolal, Y., Rarey, J., and Ramjugernath, D.:
                     Estimation of pure component properties. Part 3.
                     Estimation of the vapor pressure of non-electrolyte
                     organic compounds via group contributions and group
                     interactions, Fluid Phase Equilibr., 269, 117–133, 2008
    .. [MYRDAL1997]  Myrdal, P. B. and Yalkowsky, S. H.: Estimating pure
                     component vapor pressures of complex organic molecules,
                     Ind. Eng. Chem. Res., 36, 2494–2499, 1997.
    .. [STEIN1994]   Stein, S. E. and Brown, R. L.: Estimation of normal
                     boiling points from group contributions,
                     J. Chem. Inf. Comp. Sci., 34, 581–587, 1994.
    .. [JOBACK1987]  Joback, K. G. and Reid, R. C.: Estimation of
                     pure-component properties from group-contributions,
                     Chem. Eng. Commun., 57, 233– 243, 1987.
    .. [AIOMFAC]     Zuend, A., Marcolli, C., Luo, B. P., and Peter, T.: A
                     thermodynamic model of mixed organic-inorganic aerosols to
                     predict activity coefficients, Atmos. Chem. Phys., 8,
                     4559-4593, doi:10.5194/acp-8-4559-2008, 2008.
    .. [COMP2011]    Compernolle, S., Ceulemans, K., and Müller, J.-F.:
                     EVAPORATION: a new vapour pressure estimation method
                     for organic molecules including non-additivity and
                     intramolecular interactions, Atmos. Chem. Phys.,
                     11, 9431-9450, doi:10.5194/acp-11-9431-2011, 2011.
     """

    vapour_pressure = {
        'nannoolal':   vapour_pressures.nannoolal,
        'myrdal':      vapour_pressures.myrdal_and_yalkowsky,
        # Evaporation doesn't use boiling point
        'evaporation': lambda c, t, b: vapour_pressures.evaporation(c, t),
        }[vp_method]
    boiling_point = {
        'joback_and_reid': boiling_points.joback_and_reid,
        'stein_and_brown': boiling_points.stein_and_brown,
        'nannoolal':       boiling_points.nannoolal,
        }[bp_method]
    ideality = interactions_method #== 'ideal'

    abundances = {}
    coefficients = {}
    totals = {}
    for temperature in temperatures:
        pressure_data = {
            compound: vapour_pressure(compound, temperature, boiling_point(compound))
            for compound in organic_compounds.keys()
            }
        for humidity in humidities:
            water_result, compounds_result = partition_models.partition_model_org(
                organic_compounds, pressure_data,
                soluble_core, temperature, humidity, ideality)
            for result in compounds_result:
                abundances[(temperature, humidity), str(result.compound).strip()] = result.condensed_abundance
                coefficients[(temperature, humidity), str(result.compound).strip()] = result.activity_coefficient
            totals[(temperature, humidity), 'Quantity'] = sum(c.condensed_abundance * c.molar_mass for c in compounds_result)
            totals[(temperature, humidity), 'Water activity'] = water_result.activity

    return Result(
        Table(
            'abundances',
            title='Condensed phase abundance (µmol/m³) of secondary organic material (SOA)',
            rows_title=('Temperature', 'Humidity'), rows_unit=('K', '%'), rows=product(temperatures, humidities),
            cols_title='Compound', cols=[str(c).strip() for c in organic_compounds],
            data=abundances,
            ),
        Table(
            'coefficients',
            title='Activity coefficients (unitless) of secondary organic material (SOA)',
            rows_title=('Temperature', 'Humidity'), rows_unit=('K', '%'), rows=product(temperatures, humidities),
            cols_title='Compound', cols=[str(c).strip() for c in organic_compounds],
            data=coefficients,
            ),
        Table(
            'totals',
            title='Total quantities (µg/m³) and water activity (unitless) of secondary organic material (SOA)',
            rows_title=('Temperature', 'Humidity'), rows_unit=('K', '%'), rows=product(temperatures, humidities),
            cols_title='Result', cols=('Quantity', 'Water activity'),
            data=totals,
            )
        )
