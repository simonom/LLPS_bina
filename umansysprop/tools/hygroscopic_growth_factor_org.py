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
Hygroscopic growth factors [Organic systems]
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
from .. import hygroscopicity
from .. import critical_properties
from .. import liquid_densities

class HandlerForm(Form):
    organic_compounds = SMILESDictField(
        'Organic compounds', entry_label='SMILES', data_label='moles',
        compounds=[
            ('C(CC(=O)O)C(=O)O',                 'Succinic acid'),
            ('C(=O)(C(=O)O)O',                   'Oxalic acid'),
            ('O=C(O)CC(O)=O',                    'Malonic acid'),
            ('CCCCC/C=C/C/C=C/CC/C=C/CCCC(=O)O', 'Pinolenic acid'),
            ])
    interactions_method = SelectField(
        'Interaction model (please note calculating activity coefficients can significantly increase model simulation time)', default='nonideal', choices=[
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
    size_dependance = SizedependanceField(
        'The diameter of the dry, spherical, particle with prescribed surface tension of the aqueous droplet'
        , validators=[InputRequired()])
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
    temperatures = FloatRangeField(
        'Temperatures (K)', default=298.15, validators=[
            NumberRange(min=173.15, max=400.0),
            Length(min=1, max=100, message='Temperatures must have between 1 and 100 values'),
            ])
    humidities = FloatRangeField(
        'Humidities (%)', default=50.0, validators=[
            NumberRange(min=50.0, max=100.0),
            Length(min=1, max=20, message='Relative humidities must have between 1 and 20 values'),
            ])


def handler(
        organic_compounds, interactions_method, vp_method,
        bp_method, size_dependance, density_method, properties_method,
        temperatures, humidities):
    """
    Calculates the hygroscopic growth factor of an inorganic/organic mixture
    from a prescribed total condensed phase concentration, existing core and
    ambient conditions (all given *temperatures* and relative *humidities*).
    The *vp_method* parameter is one of the strings:

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

    The density method calculates the sub-cooled density for all specified
    *organic_compounds* (given as a sequence of SMILES strings) at all given
    *temperatures* (a sequence of floating point values giving temperatues in
    degrees Kelvin).  The *density_method* parameter is one of the strings:

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

    The result is a single table with the temperatures (K), humidity (%),
    growth factor (unitless), Kappa Kohler value (unitless) and mass increase
    (unitless).

    .. [AIOMFAC]     Zuend, A., Marcolli, C., Luo, B. P., and Peter, T.: A
                     thermodynamic model of mixed organic-inorganic aerosols to
                     predict activity coefficients, Atmos. Chem. Phys., 8,
                     4559-4593, doi:10.5194/acp-8-4559-2008, 2008.
    .. [GIRO1994]    Girolami, G.S., Journal of Chemical Education, 1994, 71,
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
        'joback':      boiling_points.joback_and_reid,
        'stein':       boiling_points.stein_and_brown,
        'nannoolal':   boiling_points.nannoolal,
        }[bp_method]
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
    ideality = interactions_method #== 'ideal'
    #diameter, surface_tension =

    equilib_vp = {}
    totals = {}
    equilib_vp = {}
    act_coeffs = {}
    for temperature in temperatures:
        pressure_data = {
            compound: vapour_pressure(compound, temperature, boiling_point(compound))
            for compound in organic_compounds.keys()
            }
        density_data = {
            compound: liquid_density(compound, temperature, critical_property(compound, boiling_point(compound)))
            for compound in organic_compounds.keys()
            }
        for humidity in humidities:
            growth_factor, Kappa, water_activity, vp, coeffs, mass_frac_sol = hygroscopicity.growth_factor_model_org(
                organic_compounds, pressure_data, density_data, temperature, humidity, ideality, size_dependance)
            for compound, result in vp.items():
                equilib_vp[(temperature, humidity), str(compound).strip()] = result
            for compound, result in coeffs.items():
                act_coeffs[(temperature, humidity), str(compound).strip()] = result
            #totals[(temperature, humidity), 'Water activity'] = water_activity
            totals[(temperature, humidity), 'Growth factor'] = growth_factor
            totals[(temperature, humidity), 'Kappa'] = Kappa
            totals[(temperature, humidity), 'Mass Increase'] = mass_frac_sol

    return Result(
        Table(
            'totals',
            title='Growth factor (unitless) and Kappa value (unitless) of aerosol particle',
            rows_title=('Temperature', 'Humidity'), rows_unit=('K', '%'), rows=product(temperatures, humidities),
            cols_title='Result', cols=('Growth factor', 'Kappa','Mass Increase'),
            data=totals,
            ),
        Table(
            'equilib_vp',
            title='Equilibrium vapour pressure (log10 (atm)) of each organic compound in the liquid phase',
            rows_title=('Temperature', 'Humidity'), rows_unit=('K', '%'), rows=product(temperatures, humidities),
            cols_title='Compound', cols=[str(c).strip() for c in organic_compounds],
            data=equilib_vp,
            ),
        Table(
            'act_coeff',
            title='Activity coefficients (unitless) of each organic compound in the liquid phase',
            rows_title=('Temperature', 'Humidity'), rows_unit=('K', '%'), rows=product(temperatures, humidities),
            cols_title='Compound', cols=[str(c).strip() for c in organic_compounds],
            data=act_coeffs,
            )
        )

