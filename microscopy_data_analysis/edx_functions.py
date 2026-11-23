from itertools import cycle
from pathlib import Path

import h5py
import hyperspy.api as hs
import matplotlib.pyplot as plt
import numpy as np
import exspy as exs
from matplotlib.patches import Rectangle
from matplotlib.legend import Legend
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import UnivariateSpline


_ELEMENT_SYMBOLS = {
    properties["General_properties"]["Z"]: symbol
    for symbol, properties in exs.material.elements.as_dictionary().items()
    if isinstance(properties, dict) and "General_properties" in properties
}


def auto_h5path(filepath,verbose=True):
    """Return the internal HDF5 path reached by following single-child groups.

    Start at the root of `filepath` and stop when the current group has zero
    or multiple children. Print the resulting path when `verbose` is true.
    """
    with h5py.File(filepath,'r') as h5file:
        h5path="./"
        keys=list(h5file[h5path].keys())
        while len(keys)==1:
            h5path +=keys[0]+"/"
            keys=list(h5file[h5path].keys())
    if verbose:
        print(h5path)
    return h5path
            
def show_subdirectories(filepath,h5path):
    """Print the children of `h5path` and their immediate children.

    Open the HDF5 file at `filepath` in read-only mode and indent the second
    level. Return None.
    """
    with h5py.File(filepath,'r') as h5file:
        for i in h5file[h5path]:
            print(i)
            for j in h5file[h5path][i]:
                print("   "+j)

def areas_info(filepath,h5path,measurements):
    """List areas and classify their measurements by name prefix.

    Search the HDF5 group at `h5path` in `filepath`. The `measurements`
    mapping supplies name prefixes and their corresponding type codes.
    Return three parallel lists containing area names, measurement names
    per area, and matching type codes per area.
    """
    with h5py.File(filepath,'r') as h5file:
        keys=list(h5file[h5path].keys())
        keys=[i for i in keys if i[:4]=="Area"]
        key_props=[[] for i in keys]
        measurement_type=[[] for i in keys]

        for c,i in enumerate(keys):
            subkeys=list(h5file[h5path+"/"+i].keys())
            for k in subkeys:
                for j in measurements:
                    if k[:len(j)]==j:
                        key_props[c].append(k)
                        measurement_type[c].append(measurements[j])
    return keys,key_props,measurement_type


def extract_data(filepath,verbose=True):
    """Load an EDAX HDF5 file into a flat, alphabetically ordered dictionary.

    Keys combine the zero-padded area number and spatial dimension, as in
    `Area 010 spatial 1D`. Each entry includes the area image, field size,
    and measurement names, type codes, and regions for that group. The 0D
    entries include spectra. The 1D and 2D entries include the LineScan and
    Live Map reader outputs, respectively. All entries include per-measurement
    metadata, `BeamKV`, `spectrum`, `weight_fractions`, and
    `atomic_percentages` in measurement order. The composition values
    align with `composition_elements` in every spatial dimension.

    Always print the discovered internal HDF5 path. With `verbose=True`,
    also print the measurement names and type codes for each area.
    """
    h5path=auto_h5path(filepath)
    measurement_dict={}
    measurement_dict["Full Area"]=0
    measurement_dict["Selected Area"]=1
    measurement_dict["EDS Spot"]=2
    measurement_dict["Free Draw"]=3
    measurement_dict["LineScan"]=4
    measurement_dict["Live Map"]=5
    
    keys,key_props,mt=areas_info(filepath,h5path,measurement_dict)
    
    if verbose:
        for c, area in enumerate(keys):
            print(area, key_props[c], mt[c])

    experiment = {}
    spatial_groups = (
        ("0D", (0, 1, 2, 3), get_0D_spectrum),
        ("1D", (4,), get_1D_linescan),
        ("2D", (5,), get_2D_map),
    )
    for c, area in enumerate(keys):
        area_path = h5path.rstrip("/") + "/" + area
        img, size = get_image(filepath, area_path)
        area_number = area[5:]
        area_label = f"Area {int(area_number):03d}" if area_number.isdecimal() else area

        for dimension, kinds, read_measurements in spatial_groups:
            group = [
                (name, kind)
                for name, kind in zip(key_props[c], mt[c])
                if kind in kinds
            ]
            if not group:
                continue
            measurements, measurement_types = map(list, zip(*group))
            entry = {
                "image": img,
                "image_width_height_mm": size,
                "measurements": measurements,
                "_measurement_types": measurement_types,
                "regions_pix": np.array(
                    get_region(filepath, area_path, measurement_types, measurements, img=img)
                ),
            }
            result = read_measurements(filepath, area_path, measurements)
            if dimension == "0D":
                fields = ("spectrum", "metadata", "composition_elements", "weight_fractions", "atomic_percentages")
            else:
                fields = (
                    "elements", "element_maps", "spectrum", "dataset",
                    "metadata", "total_counts_image", "se_image", "composition_elements",
                    "weight_fractions", "atomic_percentages",
                )
            entry.update(zip(fields, result))
            entry["BeamKV"] = [measurement["BeamKV"] for measurement in entry["metadata"]]
            experiment[f"{area_label} spatial {dimension}"] = entry

    return dict(sorted(experiment.items()))
        
def get_region(filepath,h5path,mtypes,measurements,img=None):
    """Return regions for measurements in an HDF5 area.

    Use `h5path` as the area path in `filepath` and pair each measurement
    name with its code in `mtypes`. Use the supplied area image `img`,
    or load it when omitted. Codes 0 through 3 return pixel coordinates
    as `[x, y, width, height]`, with zero extents for EDS Spot and Free Draw.
    LineScan returns the stored endpoints, line width, point count, and
    step without conversion. Live Maps contribute no region entry.
    """
    if img is None:
        img, _ = get_image(filepath, h5path)
    regions=[] # x,y,x_width,y_height
    with h5py.File(filepath,'r') as h5file:
        for c,i in enumerate(measurements):

            if mtypes[c]==0:#Full Area
                regions.append([0,0,img.shape[1],img.shape[0]])
            elif mtypes[c]==1:#Selected Area
                coord=h5file[h5path+'/'+i+'/REGION'][0]
                width_pos=coord[0]*img.shape[1]
                height_pos=coord[1]*img.shape[0]
                x_width=coord[2]*img.shape[1]
                y_height=coord[3]*img.shape[0]
                regions.append([width_pos,height_pos,x_width,y_height])
            elif mtypes[c] == 2 or mtypes[c] == 3:#EDS Spot or Free Draw
                coord=h5file[h5path+'/'+i+'/REGION'][0]
                width_pos=coord[0]*img.shape[1]
                height_pos=coord[1]*img.shape[0]
                regions.append([width_pos,height_pos,0,0])
            elif mtypes[c]==4: #LineScan
                # x1,y1,x2,y2,Linewidth,NPoints,Step
                region=h5file[h5path+'/'+i+'/REGION']
                x1=region["X1"]
                x2=region["X2"]
                y1=region["Y1"]
                y2=region["Y2"]
                lw=region["LineWidth"]
                npoints=region["NPoints"]
                step=region["Step"]
                regions.append([x1,y1,x2,y2,lw,npoints,step])

            # Live Map has no field "REGION"

    return regions
        
                
def get_image(filepath,h5path,micro_precision=1,plotting=False ):
    """Return the field-of-view image and its physical dimensions.

    Read and reshape `FOVIMAGE` from the area at `h5path` in `filepath`.
    Return `(image, [width_mm, height_mm])` by matching the longer physical
    dimension to the longer image axis. For square images, use the stored
    width and height field names.
    If `plotting` is true, draw the image in grayscale on the current axes
    and round the micrometer labels to `micro_precision` decimal places.
    """
    with h5py.File(filepath,'r') as h5file:
        params = h5file[h5path + "/FOVIMAGECOLLECTIONPARAMS"][0]
        img = h5file[h5path + "/FOVIMAGE"][:].reshape(params[1], params[2])
        width_mm = params["mmFieldWidth"]
        height_mm = params["mmFieldHeight"]

    pixel_height, pixel_width = img.shape
    if ((pixel_width > pixel_height and width_mm < height_mm)
            or (pixel_width < pixel_height and width_mm > height_mm)):
        width_mm, height_mm = height_mm, width_mm

    if plotting:
        plt.imshow(img, cmap="gray")
        plt.title(h5path)
        plt.xlabel(f"{width_mm * 1000:.{micro_precision}f} µm")
        plt.ylabel(f"{height_mm * 1000:.{micro_precision}f} µm")
        plt.xticks([])
        plt.yticks([])
    return img, [width_mm, height_mm]


def _plot_area_image(entry, area, digits_smaller_microns, indices):
    """Draw the field of view and title it with the selected beam voltages."""
    img = entry["image"]
    height, width = img.shape[:2]
    _, ax = plt.subplots()
    ax.imshow(img, cmap="gray", origin="upper", extent=(0, width, height, 0))
    field_width, field_height = entry["image_width_height_mm"]
    area_label = area.split(" spatial ", 1)[0]
    voltages = dict.fromkeys(entry["BeamKV"][index] for index in indices)
    voltage_label = ", ".join(f"{voltage:g} kV" for voltage in voltages)
    ax.set(
        xlim=(0, width), ylim=(height, 0), xticks=[], yticks=[],
        xlabel=f"{field_width * 1000:.{digits_smaller_microns}f} µm",
        ylabel=f"{field_height * 1000:.{digits_smaller_microns}f} µm",
        title=f"{area_label} at {voltage_label}" if voltages else area_label,
    )
    return ax


def _unmasked_indices(count, mask):
    """Return the indices of measurements whose mask entries are false."""
    if mask is None:
        return list(range(count))
    if len(mask) != count:
        raise ValueError("Provide one mask value per measurement.")
    return [index for index, hidden in enumerate(mask) if not hidden]


def _measurement_styles(entry, count, mpl_colorlist, name_list):
    """Pair each measurement label with a color from the chosen cycle."""
    names = entry["measurements"] if name_list is None else list(name_list)
    if len(names) != count:
        raise ValueError("Provide one legend label per measurement.")
    return list(zip(names, cycle(_plot_colors(mpl_colorlist))))


def _plot_colors(mpl_colorlist):
    """Return the supplied colors or the Matplotlib default color cycle."""
    colors = (
        plt.rcParams["axes.prop_cycle"].by_key()["color"]
        if mpl_colorlist is None else list(mpl_colorlist)
    )
    if not colors:
        raise ValueError("mpl_colorlist must contain at least one color.")
    return colors


def _place_plot_legends(ax, legend_outside=True, element_handles=None):
    """Place measurement labels first and optional element labels below them."""
    ax.figure.set_layout_engine("constrained")
    if element_handles is None:
        handles, labels = ax.get_legend_handles_labels()
        primary = ax.legend(
            handles, labels, loc="upper left" if legend_outside else "upper right",
            bbox_to_anchor=(1.02, 1) if legend_outside else None,
        )
        ax._edx_primary_legend = primary
        return

    primary = getattr(ax, "_edx_primary_legend", ax.get_legend())
    previous = getattr(ax, "_edx_element_legend", None)
    if previous is not None:
        previous.remove()
    if primary is not None:
        primary.set_loc("upper left" if legend_outside else "upper right")
        primary.set_bbox_to_anchor((1.02, 1) if legend_outside else (1, 1))
    secondary = Legend(
        ax, element_handles, [handle.get_label() for handle in element_handles], loc="upper left",
        bbox_to_anchor=(1.02, 1) if legend_outside else None,
    )
    ax.add_artist(secondary)
    ax._edx_element_legend = secondary
    if legend_outside and primary is not None:
        ax.figure.canvas.draw()
        bottom = ax.transAxes.inverted().transform(
            primary.get_window_extent(ax.figure.canvas.get_renderer()).corners()[0]
        )[1]
        secondary.set_bbox_to_anchor((1.02, bottom - 0.025))


def plot_0D_image(experiment, area, mpl_colorlist=None, name_list=None,
                  digits_smaller_microns=1, mask=None, legend_outside=True):
    """Plot the field-of-view image and measurement regions for a 0D entry.

    Pass the dictionary from `extract_data` and a key such as
    `Area 016 spatial 0D`. Legend labels default to the stored measurement
    names. Provide one label per measurement to override them with
    `name_list`. Colors cycle through `mpl_colorlist` or the Matplotlib
    default color cycle. Use `digits_smaller_microns` decimal places for
    the field dimensions. Skip measurements whose mask entries are true.
    The title includes the visible measurements' beam voltages.
    Return the axes.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    entry = experiment[area]
    regions = entry["regions_pix"]
    styles = _measurement_styles(entry, len(regions), mpl_colorlist, name_list)
    indices = _unmasked_indices(len(regions), mask)
    ax = _plot_area_image(entry, area, digits_smaller_microns, indices)
    height, width = entry["image"].shape[:2]
    for index in indices:
        x, y, region_width, region_height = regions[index]
        name, color = styles[index]
        if region_width == 0 and region_height == 0:
            ax.plot(x, y, "o", label=name, color=color)
        else:
            full_area = (x, y, region_width, region_height) == (0, 0, width, height)
            ax.add_patch(Rectangle(
                (x, y), region_width, region_height, fill=False,
                edgecolor=color, label=name, clip_on=not full_area,
                zorder=3 if full_area else 1,
            ))

    if indices:
        _place_plot_legends(ax, legend_outside)
    return ax


def plot_1D_image(experiment, area, digits_smaller_microns=1, mask=None, legend_outside=True):
    """Plot LineScan endpoints and dashed paths on the field-of-view image.

    Pass the dictionary from `extract_data` and a key such as
    `Area 005 spatial 1D`. Convert the stored fractional endpoints to
    image coordinates and mark the start and end with different colors.
    Use `digits_smaller_microns` decimal places for the field dimensions.
    Skip LineScans whose mask entries are true. The title includes the
    visible LineScans' beam voltages. Return the axes.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    entry = experiment[area]
    height, width = entry["image"].shape[:2]
    regions = entry["regions_pix"]
    indices = _unmasked_indices(len(regions), mask)
    ax = _plot_area_image(entry, area, digits_smaller_microns, indices)
    endpoints = np.asarray([regions[index][:4] for index in indices]).reshape(-1, 4)
    endpoints = endpoints * [width, height, width, height]
    for (x1, y1, x2, y2), index in zip(endpoints, indices):
        ax.plot([x1, x2], [y1, y2], "--", label=entry["measurements"][index])
    if len(endpoints):
        x_start, y_start, x_end, y_end = endpoints.T
        ax.scatter(x_start, y_start, color="tab:green", label="Start", zorder=3)
        ax.scatter(x_end, y_end, color="tab:red", label="End", zorder=3)
        _place_plot_legends(ax, legend_outside)
    return ax


def plot_2D_image(experiment, area, index=0, digits_smaller_microns=1):
    """Plot the grayscale SE overview and the selected Live Map's beam voltage.

    Use `digits_smaller_microns` decimal places for the field dimensions.
    Return the axes.
    """
    ax = _plot_area_image(experiment[area], area, digits_smaller_microns, [index])
    ax.set_title(f"{ax.get_title()} SE-Signal")
    ax.figure.tight_layout()
    return ax


def plot_1D_maps(experiment, area, index=0, mask=None, mpl_colorlist=None,
                 logscale_y=False, legend_outside=True):
    """Plot elemental profiles and two image strips for one LineScan.

    Select the scan with `index` and hide element symbols listed in `mask`.
    Colors cycle through `mpl_colorlist` or the Matplotlib defaults.
    Plot distance from the start in micrometers. The grayscale SE and
    plasma X-ray counts strips each occupy 10 percent of the y-range
    below zero. With logscale_y=True, use a positive count axis starting at
    0.5 and a separate strip axis below it. Return the profile axes.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    if mask is None:
        mask = []
    entry = experiment[area]
    cps = np.asarray(entry["total_counts_image"][index]).reshape(1, -1)
    se_image = np.asarray(entry["se_image"][index]).reshape(1, -1)
    orbitals = entry["metadata"][index]["orbitals"]
    masked_elements = set(mask)
    x1, y1, x2, y2 = np.asarray(entry["regions_pix"][index][:4]).reshape(4)
    field_width, field_height = entry["image_width_height_mm"]
    distance_um = 1000 * np.hypot((x2 - x1) * field_width, (y2 - y1) * field_height)

    fig = plt.figure(layout="constrained")
    if logscale_y:
        grid = fig.add_gridspec(2, 1, height_ratios=(4, 1), hspace=0)
        ax = fig.add_subplot(grid[0])
        strip_ax = fig.add_subplot(grid[1], sharex=ax)
        ax.tick_params(axis="x", labelbottom=False)
        strip_ax.set_yticks([])
    else:
        ax = fig.add_subplot()
        strip_ax = ax
    peak_count = 0
    for profile, element, orbital, color in zip(
        entry["element_maps"][index], entry["elements"][index], orbitals,
        cycle(_plot_colors(mpl_colorlist)),
    ):
        if element not in masked_elements:
            counts = np.asarray(profile).ravel()
            distances = np.linspace(0, distance_um, counts.size)
            ax.plot(distances, counts, color=color, label=f"{element} {orbital}")
            peak_count = max(peak_count, np.max(counts, initial=0))

    top = max(1, 1.05 * peak_count) if logscale_y else (1.05 * peak_count if peak_count > 0 else 1)
    stripe_height = 1 if logscale_y else top / 8
    for image, cmap, bottom in (
        (se_image, "gray", -2 * stripe_height),
        (cps, "plasma", -stripe_height),
    ):
        if not image.size:
            continue
        half_step = distance_um / (2 * (image.size - 1)) if image.size > 1 else 0
        left = -half_step if distance_um else -0.5
        right = distance_um + half_step if distance_um else 0.5
        strip = strip_ax.imshow(
            image, cmap=cmap, aspect="auto", interpolation="nearest", zorder=0,
            extent=(left, right, bottom, bottom + stripe_height),
        )
        if cmap == "plasma":
            colorbar = fig.colorbar(strip, ax=[ax, strip_ax] if logscale_y else ax, label="X-ray Counts")
            colorbar.ax.set_xlabel("Greyscale\nSE-Image")

    area_label = area.split(" spatial ", 1)[0]
    ax.set(title=f"{area_label} at {entry['BeamKV'][index]:g} kV",
           ylabel="Counts", ylim=(0.5 if logscale_y else -2 * stripe_height, top))
    strip_ax.set_xlabel("Distance from start (µm)")
    if logscale_y:
        ax.set_yscale("log", nonpositive="mask")
        strip_ax.set_ylim(-2, 0)
    ax.set_xlim((0, distance_um) if distance_um else (-0.5, 0.5))
    if not distance_um:
        ax.set_xticks([0])
    if not logscale_y:
        ticks = MaxNLocator(nbins=6).tick_values(0, top)
        ax.set_yticks(ticks[(0 <= ticks) & (ticks <= top)])
    if ax.lines:
        _place_plot_legends(ax, legend_outside)
    return ax


def _element_map_colors(count):
    """Return RGBYCM first, then colors spaced apart in RGB space."""
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
              (1, 1, 0), (0, 1, 1), (1, 0, 1)]
    if count > len(colors):
        levels = np.linspace(0, 1, 11)
        candidates = np.array(np.meshgrid(levels, levels, levels)).reshape(3, -1).T
        candidates = candidates[candidates.max(axis=1) >= 0.5]
        distances = np.min(
            np.sum((candidates[:, None] - colors) ** 2, axis=2), axis=1
        )
        while len(colors) < count:
            color = candidates[np.argmax(distances)]
            colors.append(color)
            distances = np.minimum(distances, np.sum((candidates - color) ** 2, axis=1))
    return np.asarray(colors[:count])


def plot_2D_maps(experiment, area, index=0, mask=None, grid_division=3):
    """Plot the SE image, X-ray counts, overlay, and element maps for one Live Map.

    Use `index` to select the Live Map within a spatial 2D experiment entry.
    Scale each element map from zero to its own peak and use that scale as
    color intensity on black. The overlay sums those RGB contributions
    and clips each channel to its display range. List element symbols in
    `mask` to omit them from both the overlay and the individual panels.
    Remaining elements retain their original colors. A `grid_division`
    above 1 divides each image axis into that many equal parts.
    Return the figure.
    """
    if mask is None:
        mask = []
    if not isinstance(grid_division, (int, np.integer)) or grid_division < 1:
        raise ValueError("grid_division must be a positive integer.")

    entry = experiment[area]
    se_image = entry["se_image"][index]
    total_counts_image = entry["total_counts_image"][index]
    element_maps = entry["element_maps"][index]
    elements = entry["elements"][index]
    orbitals = entry["metadata"][index]["orbitals"]
    masked_elements = set(mask)
    visible_maps = [
        (image, element, orbital, color)
        for image, element, orbital, color in zip(
            element_maps, elements, orbitals, _element_map_colors(len(element_maps))
        )
        if element not in masked_elements
    ]

    rows = 1 + (len(visible_maps) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(10, 3 * rows), squeeze=False,
                            layout="constrained")
    area_label = area.split(" spatial ", 1)[0]
    fig.suptitle(f"{area_label} at {entry['BeamKV'][index]:g} kV")
    for ax in axes.flat:
        ax.set_axis_off()

    axes[0, 0].imshow(se_image, cmap="gray")
    axes[0, 0].set_title("SE-Signal")
    counts = axes[0, 2].imshow(total_counts_image, cmap="plasma")
    axes[0, 2].set_title("X-ray Counts")
    fig.colorbar(counts, ax=axes[0, 2])

    overlay = np.zeros((*total_counts_image.shape, 3))
    for ax, (image, element, orbital, color) in zip(axes.flat[3:], visible_maps):
        intensity = np.where(np.isfinite(image) & (image > 0), image, 0)
        peak = intensity.max(initial=0)
        rgb = (intensity / (peak if peak > 0 else 1))[..., None] * color
        overlay += rgb
        ax.imshow(rgb)
        ax.set_title(f"{element} {orbital}")

    axes[0, 1].imshow(np.clip(overlay, 0, 1))
    axes[0, 1].set_title("Overlay")
    for ax in axes.flat[:3 + len(visible_maps)]:
        for division in range(1, grid_division):
            position = division / grid_division
            ax.plot([position, position], [0, 1], transform=ax.transAxes,
                    color="white", linestyle="--", linewidth=0.8)
            ax.plot([0, 1], [position, position], transform=ax.transAxes,
                    color="white", linestyle="--", linewidth=0.8)
    return fig


def _spectrum_normalization_factors(spectra, normalize):
    """Return scale factors for the spectra selected for plotting."""
    if normalize not in ("none", "integrated", "maximum"):
        raise ValueError("normalize must be 'none', 'integrated', or 'maximum'.")
    factors = np.ones(len(spectra))
    if normalize == "none" or len(spectra) < 2:
        return factors

    if normalize == "integrated":
        values = np.array([np.sum(counts, dtype=float) for _, counts in spectra])
        target = values.max()
    else:
        peaks = [np.max(counts, initial=0) for _, counts in spectra]
        reference = np.argmax(peaks)
        target = peaks[reference]
        if target == 0:
            return factors
        energies, counts = spectra[reference]
        peak_energy = energies[np.argmax(counts)]
        values = np.array([
            np.interp(peak_energy, energies, counts, left=0, right=0)
            if len(energies) else 0
            for energies, counts in spectra
        ])

    np.divide(target, values, out=factors, where=values > 0)
    return factors


def _set_spectrum_y_axis(ax, peak_count, logscale_y_minimum):
    """Set count limits and apply a logarithmic scale for a positive minimum."""
    upper_limit = max(1, 1.1 * peak_count, 1.1 * logscale_y_minimum)
    if logscale_y_minimum == 0 and peak_count > 0:
        upper_limit = 1.1 * peak_count
    ax.set_ylim(logscale_y_minimum, upper_limit)
    if logscale_y_minimum > 0:
        ax.set_yscale("log", nonpositive="mask")


def plot_0D_spectra(experiment, area, mpl_colorlist=None, name_list=None, mask=None,
                    normalize="none", logscale_y_minimum=0, legend_outside=True):
    """Plot the energy and counts spectra for a 0D experiment entry.

    Legend labels and colors use the same options as `plot_0D_image`.
    Skip spectra whose mask entries are true. With `normalize="integrated"`,
    scale curves to the largest full-spectrum count sum. With
    `normalize="maximum"`, match the highest peak at its energy, using
    interpolation for other energy grids. Curves with zero counts at
    the normalization reference retain their original scale.
    Use the highest remaining
    BeamKV in the title and as the upper energy limit, and print a warning
    if the remaining voltages differ. Set the count limits to zero and
    110 percent of the selected peak count, using 1 for a zero peak.
    A positive `logscale_y_minimum` sets the lower limit of a logarithmic
    y-axis and hides nonpositive counts. Zero uses a linear y-axis.
    An empty selection yields empty axes with the area name as the title.
    Return the axes.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    if not np.isfinite(logscale_y_minimum) or logscale_y_minimum < 0:
        raise ValueError("logscale_y_minimum must be finite and nonnegative.")
    entry = experiment[area]
    spectra = entry["spectrum"]
    indices = _unmasked_indices(len(spectra), mask)
    styles = _measurement_styles(entry, len(spectra), mpl_colorlist, name_list)
    selected = [spectra[index] for index in indices]
    factors = _spectrum_normalization_factors(selected, normalize)

    _, ax = plt.subplots()
    area_label = area.split(" spatial ", 1)[0]
    ax.set(xlabel="Photon energy (keV)", ylabel="Counts", title=area_label)
    peak_count = 0
    for index, (energies, counts), factor in zip(indices, selected, factors):
        if factor != 1:
            counts = np.asarray(counts) * factor
        peak_count = max(peak_count, np.max(counts, initial=0))
        name, color = styles[index]
        ax.plot(energies, counts, label=name, color=color)

    if indices:
        voltages = {entry["BeamKV"][index] for index in indices}
        if len(voltages) > 1:
            print("Mutltiple BeamKV values present!")
        beam_kv = max(voltages)
        ax.set(title=f"{area_label} at {beam_kv:g} kV", xlim=(0, beam_kv))
        _place_plot_legends(ax, legend_outside)
    if indices or logscale_y_minimum > 0:
        _set_spectrum_y_axis(ax, peak_count, logscale_y_minimum)
    return ax

def get_x_axis(spectrum):
    """Return the channel midpoint energies in keV for a spectrum."""
    electron_volts_per_channel=10
    energies = (np.arange(spectrum.shape[-1])+0.5) * electron_volts_per_channel / 1000 
    # Technically the detector has 4096 bins covering the range from 0 to 40.950 keV.
    # However, it is adaptively saved only up to the highest energy with non-zero occuring counts.
    # Here, energies represents mid-bin positions in keV.
    return energies


def plot_spectrum_from_map(experiment, area, index=0, logscale_y=False, legend_outside=True):
    """Plot the average and peak-location spectra from a LineScan or Live Map.

    Select the dataset with `index`. The final axis contains energy
    channels. Average over the other axes and use the full spectrum from
    the position of the highest individual count for the Max Counts curve.
    Use the selected measurement's beam voltage as the upper energy limit.
    With `logscale_y=True`, use the smallest positive average count as the
    lower limit, or 0.5 counts if the average has no positive values.
    Return the axes.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    entry = experiment[area]
    dataset = np.asarray(entry["dataset"][index])
    if not dataset.size:
        raise ValueError("The selected dataset contains no counts.")

    max_index = np.unravel_index(np.argmax(dataset), dataset.shape)
    max_counts = dataset[max_index[:-1]]
    average = dataset.mean(axis=tuple(range(dataset.ndim - 1)))
    energies = get_x_axis(max_counts)
    beam_kv = entry["BeamKV"][index]
    lower_limit = 0
    if logscale_y:
        positive_counts = average[average > 0]
        lower_limit = positive_counts.min() if positive_counts.size else 0.5
    fig, ax = plt.subplots()
    ax.plot(energies, max_counts, label="Max Counts")
    ax.plot(energies, average, label="Average")
    area_label = area.split(" spatial ", 1)[0]
    ax.set(title=f"{area_label} at {beam_kv:g} kV", xlim=(0, beam_kv),
           xlabel="Photon energy (keV)", ylabel="Counts")
    _set_spectrum_y_axis(ax, dataset[max_index], lower_limit)
    _place_plot_legends(ax, legend_outside)
    return ax


def _snip_background(counts, half_window):
    """Estimate a nonnegative background using LLS-transformed SNIP clipping."""
    baseline = np.log1p(np.log1p(np.sqrt(counts + 1)))
    for offset in range(1, min(half_window, (len(counts) - 1) // 2) + 1):
        neighbors = (baseline[:-2 * offset] + baseline[2 * offset:]) / 2
        baseline[offset:-offset] = np.minimum(baseline[offset:-offset], neighbors)
    return np.maximum((np.expm1(np.expm1(baseline))) ** 2 - 1, 0)


def _spectrum_fit_metadata(model, energy, counts, fitted, background,
                           detector_resolution, beam_voltage, method, options, selected,
                           force_positive_peaks):
    """Collect every Gaussian line component and the arrays needed to plot a fit."""
    peaks = []
    channel_width = energy[1] - energy[0]
    for component in model:
        if component in model.background_components:
            continue
        element, line = component.name.split("_", 1)
        centre = float(component.centre.value)
        sigma = float(component.sigma.value)
        area = float(component.A.value)
        parent = component.A.twin
        peaks.append({
            "name": component.name,
            "element": element,
            "orbital": line[0],
            "energy_keV": centre,
            "sigma_keV": sigma,
            "fwhm_eV": 1000 * sigma * np.sqrt(8 * np.log(2)),
            "area_counts": area,
            "height_counts": area * channel_width / (sigma * np.sqrt(2 * np.pi)),
            "relative_weight": float(
                exs.material.elements[element].Atomic_properties.Xray_lines[line].weight
            ),
            "tied_to": parent.component.name if parent is not None else None,
            "amplitude_expression": component.A.twin_function_expr if parent is not None else None,
            "in_fit_range": bool(energy[selected][0] <= centre <= energy[selected][-1]),
        })
    return {
        "energy_keV": energy.copy(),
        "measured_counts": counts.copy(),
        "fitted_counts": fitted,
        "background_counts": background,
        "BeamKV": float(beam_voltage),
        "DetectorResolution": float(detector_resolution),
        "background_method": method,
        "background_options": dict(options),
        "force_positive_peaks": bool(force_positive_peaks),
        "fit_range_keV": (float(energy[selected][0]), float(energy[selected][-1])),
        "peaks": sorted(peaks, key=lambda peak: peak["energy_keV"]),
    }


def _finish_spectrum_fit(model, energy, counts, fitted, background, detector_resolution,
                         beam_voltage, method, options, selected, force_positive_peaks,
                         extrapolate_background):
    """Optionally extend the low-energy background and package the fit outputs."""
    below_fit = energy < energy[selected][0]
    if extrapolate_background and below_fit.any():
        first_channels = np.flatnonzero(selected)[:6]
        first = first_channels[0]
        slope = np.polyfit(energy[first_channels] - energy[first], background[first_channels], 1)[0]
        background[below_fit] = np.maximum(
            background[first] + slope * (energy[below_fit] - energy[first]), 0
        )
        peak_components = [component for component in model if component not in model.background_components]
        peak_counts = model.as_signal(
            component_list=peak_components, out_of_range_to_nan=False, show_progressbar=False
        ).data
        fitted[below_fit] = background[below_fit] + peak_counts[below_fit]
    metadata = _spectrum_fit_metadata(
        model, energy, counts, fitted, background, detector_resolution,
        beam_voltage, method, options, selected, force_positive_peaks,
    )
    metadata["background_extrapolation"] = "linear" if extrapolate_background else None
    metadata["extrapolated_mask"] = below_fit & extrapolate_background
    metadata["background_subtracted_counts"] = counts - background
    return fitted, background, metadata


def fit_eds_spectrum(energy, counts, elements, orbitals, detector_resolution, beam_voltage,
                     background_method="polynomial", background_options=None,
                     force_positive_peaks=False, fit_minimum_keV=0.2,
                     extrapolate_background=False):
    """Return total fitted counts, background, and a peak metadata dictionary.

    Supply energy in keV on a uniform grid, channel counts, parallel element
    and K/L/M shell lists, detector FWHM at Mn Ka in eV, and beam voltage in kV.
    Both output arrays match the input. Fit from fit_minimum_keV (default
    0.2) to the beam energy. Set fit_minimum_keV=0 to include lower channels
    directly, or extrapolate_background=True to fill channels below the
    fit with a linear background extension and the fitted peak tails.
    The extension uses the slope of the first six fitted background channels,
    stays continuous at the boundary, and is clipped at zero. Metadata marks
    extrapolated channels. Values above the fitted interval remain NaN.

    Choose a background_method and optional background_options dictionary.
    'polynomial' fits outside peak windows and accepts 'order' (default 6).
    'snip' clips peaks in log-log-square-root transformed counts and accepts
    'window_keV', the maximum half-window (default 0.4 keV).
    'spline' fits outside four-sigma peak windows with Poisson-noise weights,
    and accepts a nonnegative 'smoothing' multiplier (default 1).

    The background is held fixed during the linear least-squares peak fit.
    Set force_positive_peaks=True to use nonnegative least squares for the
    peak amplitudes, including their tied subsidiary peaks. The default
    permits negative amplitudes. SNIP and spline backgrounds are
    clipped at zero. Background settings should be checked against residuals.
    Metadata includes all Gaussian components, including tied subsidiary lines,
    with their energies, widths, fitted areas and heights, database weights,
    and amplitude ties. It also stores the arrays for plot_fitted_spectrum.
    Areas are integrated counts and heights are counts per channel.
    """
    energy = np.asarray(energy, dtype=float)
    counts = np.asarray(counts, dtype=float)
    elements, orbitals = list(elements), list(orbitals)
    if energy.ndim != 1 or counts.shape != energy.shape or energy.size < 2:
        raise ValueError("energy and counts must be matching 1D arrays with at least two channels.")
    if not np.isfinite(energy).all() or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("energy must be finite and counts must be finite and nonnegative.")
    step = energy[1] - energy[0]
    if step <= 0 or not np.allclose(np.diff(energy), step):
        raise ValueError("energy must be uniformly increasing.")
    if len(elements) != len(orbitals) or any(shell not in ("K", "L", "M") for shell in orbitals):
        raise ValueError("Provide one K, L, or M orbital per element.")
    if not np.isfinite(detector_resolution) or detector_resolution <= 0:
        raise ValueError("detector_resolution must be positive and finite.")
    if not np.isfinite(fit_minimum_keV) or fit_minimum_keV < 0:
        raise ValueError("fit_minimum_keV must be finite and nonnegative.")
    if not np.isfinite(beam_voltage) or beam_voltage <= fit_minimum_keV:
        raise ValueError("beam_voltage must be finite and greater than fit_minimum_keV.")

    option_names = {"polynomial": "order", "snip": "window_keV", "spline": "smoothing"}
    if background_method not in option_names:
        raise ValueError("background_method must be 'polynomial', 'snip', or 'spline'.")
    options = dict(background_options or {})
    peak_optimizer = "nnls" if force_positive_peaks else "lstsq"
    if options.keys() - {option_names[background_method]}:
        raise ValueError(f"Unsupported options for {background_method}.")
    fit_minimum, fit_maximum = max(fit_minimum_keV, energy[0]), min(beam_voltage, energy[-1])
    selected = (energy >= fit_minimum) & (energy <= fit_maximum)
    if selected.sum() < 8:
        raise ValueError("The fitting interval must contain at least eight channels.")

    signal = hs.signals.Signal1D(counts.copy())
    signal.set_signal_type("EDS_SEM")
    axis = signal.axes_manager.signal_axes[0]
    axis.name, axis.units = "Energy", "keV"
    axis.offset, axis.scale, axis.is_binned = float(energy[0]), float(step), True
    signal.set_microscope_parameters(
        beam_energy=float(beam_voltage), energy_resolution_MnKa=float(detector_resolution)
    )
    signal.set_lines(sorted({
        f"{element}_{orbital}a" for element, orbital in zip(elements, orbitals)
    }))
    model = signal.create_model(auto_background=False)
    model.set_signal_range_from_mask(selected)

    if background_method == "polynomial":
        order = options.get("order", 6)
        if not isinstance(order, (int, np.integer)) or not 0 <= order < selected.sum():
            raise ValueError("Polynomial order must be a nonnegative integer below the channel count.")
        model.add_polynomial_background(order=order)
        model.fit_background(start_energy=fit_minimum, end_energy=fit_maximum, optimizer="lstsq")
        model.set_signal_range_from_mask(selected)
        model.fit(optimizer=peak_optimizer, calculate_errors=not force_positive_peaks)
        fitted = model.as_signal(show_progressbar=False).data.copy()
        background = model.as_signal(
            component_list=model.background_components, show_progressbar=False
        ).data.copy()
        return _finish_spectrum_fit(
            model, energy, counts, fitted, background, detector_resolution,
            beam_voltage, background_method, options, selected, force_positive_peaks,
            extrapolate_background,
        )

    fit_energy, fit_counts = energy[selected], counts[selected]
    if background_method == "snip":
        window = options.get("window_keV", 0.4)
        if not np.isfinite(window) or window <= 0:
            raise ValueError("SNIP window_keV must be positive and finite.")
        estimated = _snip_background(fit_counts, max(1, int(np.ceil(window / step))))
    else:
        smoothing = options.get("smoothing", 1)
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("Spline smoothing must be finite and nonnegative.")
        anchors = np.ones(len(fit_energy), dtype=bool)
        for component in model:
            anchors &= np.abs(fit_energy - component.centre.value) > 4 * component.sigma.value
        if anchors.sum() < 4:
            raise ValueError("Too few peak-free channels for a cubic spline background.")
        spline = UnivariateSpline(
            fit_energy[anchors], fit_counts[anchors],
            w=1 / np.sqrt(np.maximum(fit_counts[anchors], 1)),
            s=smoothing * anchors.sum(), ext=3,
        )
        estimated = np.maximum(spline(fit_energy), 0)
    background = np.full(counts.shape, np.nan)
    background[selected] = estimated
    signal.data[selected] = fit_counts - estimated
    model.fit(optimizer=peak_optimizer, calculate_errors=not force_positive_peaks)
    fitted = model.as_signal(show_progressbar=False).data.copy() + background
    return _finish_spectrum_fit(
        model, energy, counts, fitted, background, detector_resolution,
        beam_voltage, background_method, options, selected, force_positive_peaks,
        extrapolate_background,
    )


def fit_0D_measurement(experiment, area, background_method="polynomial",
                       background_options=None, force_positive_peaks=False,
                       fit_minimum_keV=0.2, extrapolate_background=False):
    """Fit all spectra in a 0D entry and store results in its fit dictionary.

    Read the spectra, elements, shell labels, resolution, and beam voltage
    from the experiment entry. Each measurement name maps to fitted,
    background, and fit_metadata. Forward background, fit-range, and positivity options
    to fit_eds_spectrum. Return the stored dictionary after all fits succeed.
    """
    entry = experiment[area]
    fits = {}
    for index, name in enumerate(entry["measurements"]):
        energy, counts = entry["spectrum"][index]
        metadata = entry["metadata"][index]
        fitted, background, fit_metadata = fit_eds_spectrum(
            energy, counts, entry["composition_elements"][index], metadata["orbitals"],
            metadata["DetectorResolution"], entry["BeamKV"][index],
            background_method=background_method, background_options=background_options,
            force_positive_peaks=force_positive_peaks,
            fit_minimum_keV=fit_minimum_keV, extrapolate_background=extrapolate_background,
        )
        fits[name] = {
            "fitted": fitted, "background": background, "fit_metadata": fit_metadata,
        }
    entry["fit"] = fits
    return fits


def plot_fitted_spectrum(metadata, title=None, logscale_y=False, legend_outside=True):
    """Plot one measured and fitted spectrum and label all modelled peak lines.

    Pass the metadata returned by fit_eds_spectrum. Labels include subsidiary
    lines and negative fitted peaks within the fitting interval. A positive
    count lower limit is chosen automatically for a logarithmic y-axis.
    Return the single Matplotlib axes.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    energy = metadata["energy_keV"]
    measured = metadata["measured_counts"]
    fitted = metadata["fitted_counts"]
    peaks = [peak for peak in metadata["peaks"] if peak["in_fit_range"]]
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(energy, measured, label="Measured", linewidth=0.8)
    ax.plot(energy, fitted, label="Fitted", linewidth=0.9)
    finite = np.isfinite(fitted)
    peak_count = max(np.max(measured, initial=0), np.max(fitted[finite], initial=0))
    lower = 0
    if logscale_y:
        positive = measured[measured > 0]
        lower = positive.min() if positive.size else 0.5
    _set_spectrum_y_axis(ax, peak_count, lower)
    # Reserve space above the data for the line labels.
    ax.set_ylim(top=max(1, peak_count * 1.4, lower * 2))
    ax.set(
        xlim=(0, metadata["BeamKV"]), xlabel="Photon energy (keV)", ylabel="Counts",
        title=title or f"Fitted spectrum at {metadata['BeamKV']:g} kV",
    )
    for peak, position in zip(peaks, np.linspace(0.025, 0.975, len(peaks))):
        centre = peak["energy_keV"]
        y = max(lower, float(np.interp(centre, energy[finite], fitted[finite])))
        ax.annotate(
            peak["name"].replace("_", " "), xy=(centre, y),
            xytext=(position, 0.98), textcoords="axes fraction",
            rotation=90, ha="center", va="top", fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "0.6", "linewidth": 0.5},
        )
    _place_plot_legends(ax, legend_outside)
    return ax


def get_0D_spectrum(filepath,h5path,kp):
    """Read spectra, acquisition settings, and composition for measurements in `kp`.

    Resolve each name under the area at `h5path` in `filepath`. Return
    `(spectra, metadata, composition_elements, weight_fractions, atomic_percentages)`
    as five lists in input order. Each spectrum has two rows
    containing photon energies in keV and counts. Each metadata dictionary
    comes from HOSTPARAMS with SPC `DetectorResolution` (FWHM at Mn Ka,
    in eV) and `orbitals`
    from SPC ZShells, aligned with the returned composition_elements (1=K, 2=L, 3=M).
    Unrecognized or unavailable shell codes yield None. The composition values
    follow the nonzero SPC atomic numbers. Atomic percentages are NaN
    when composition_elements are present but their weight fraction is zero.
    """
    spectra=[]
    metadata = []
    composition_elements = []
    weight_fractions = []
    atomic_percentages = []
    with h5py.File(filepath,'r') as h5file:
        for name in kp:
            measurement = h5file[h5path][name]
            spc = measurement["SPC"]
            spectrum = spc["SpectrumCounts"][0]
            #ev = spc["evPch"]
            energies = get_x_axis(spectrum)
            spectra.append(np.vstack((energies, spectrum)))
            metadata.append(_get_host_metadata(measurement))
            n = int(np.asarray(spc["NumberofZElements"]).item())
            atomic_numbers = np.asarray(spc["ZAtoms"]).reshape(-1)[:n]
            weights = np.asarray(spc["WeightFraction"]).reshape(-1)[:n]
            valid = atomic_numbers != 0
            atomic_numbers = atomic_numbers[valid]
            weights = weights[valid]
            composition_elements.append([_ELEMENT_SYMBOLS[int(z)] for z in atomic_numbers])
            if "ZShells" in spc.dtype.names:
                shells = np.asarray(spc["ZShells"]).reshape(-1)[:n][valid]
            else:
                shells = np.zeros(len(atomic_numbers), dtype=int)
            metadata[-1]["orbitals"] = [
                {1: "K", 2: "L", 3: "M"}.get(int(shell)) for shell in shells
            ]
            weight_fractions.append(weights)

            masses = np.array([
                exs.material.elements[symbol].General_properties.atomic_weight
                for symbol in composition_elements[-1]
            ])
            moles = weights / masses
            total_moles = moles.sum()
            percentages = np.full(moles.shape, np.nan)
            np.divide(100 * moles, total_moles, out=percentages, where=total_moles != 0)
            atomic_percentages.append(percentages)
    return spectra, metadata, composition_elements, weight_fractions, atomic_percentages


def _get_host_metadata(measurement):
    """Read host settings and the SPC detector resolution at Mn Ka in eV."""
    hostparams = measurement["HOSTPARAMS"][()]
    field_names = {"KV": "BeamKV", "WD": "WorkingDistanceMm"}
    metadata = {
        field_names.get(field, field): hostparams[field][0]
        for field in hostparams.dtype.names
        if field not in ("AutoColCtl", "BeamCurrent", "Id")
    }
    metadata["DetectorResolution"] = float(measurement["SPC"]["DetectorResoultion"][0])
    return metadata


def get_2D_map(filepath, h5path, kp):
    """Read Live Maps named in `kp` under the area at `h5path` in `filepath`.

    Return `(elements, element_maps, total_spectrum, dataset, metadata,
    total_counts_image, se_image, composition_elements, weight_fractions,
    atomic_percentages)` in measurement order. Each map has parallel ROI
    lists of element symbols, images from .dat datasets, and orbital labels
    in its metadata dictionary. Metadata also includes a copy of the element
    list and the host and map collection parameters. `total_spectrum` is the
    spectra list from `get_0D_spectrum`, and `dataset` contains the raw SPD arrays.
    Each MAPIMAGE is reshaped to match its counts-per-second image. The
    final three lists report SPC composition in SPC element order.
    """
    kp = list(kp)
    elements = []
    element_maps = []
    dataset = []
    metadata = []
    total_counts_image = []
    se_image = []
    with h5py.File(filepath, "r") as h5file:
        area = h5file[h5path]
        for name in kp:
            live_map = area[name]
            rois = live_map["ROIs"]
            roi_names = [roi_name for roi_name in rois if roi_name.endswith(".dat")]
            roi_labels = [roi_name.rsplit(".", 1)[0].strip() for roi_name in roi_names]
            elements.append([label[:-1].strip() for label in roi_labels])
            element_maps.append([rois[roi_name][()] for roi_name in roi_names])
            dataset.append(live_map["SPD"][()])
            cps = live_map["CPS"][()]
            total_counts_image.append(cps)
            se_image.append(live_map["MAPIMAGE"][()].reshape(cps.shape))
            map_params = live_map["MAPIMAGECOLLECTIONPARAMS"][()]
            map_metadata = _get_host_metadata(live_map)
            map_metadata["elements"] = elements[-1].copy()
            map_metadata["orbitals"] = [label[-1] for label in roi_labels]
            map_metadata.update({
                field: map_params[field][0]
                for field in ("mmFieldHeight", "mmFieldWidth", "Frames", "TiltCorrected")
            })
            metadata.append(map_metadata)
    total_spectrum, _, composition_elements, weight_fractions, atomic_percentages = get_0D_spectrum(
        filepath, h5path, kp
    )
    return (
        elements, element_maps, total_spectrum, dataset, metadata, total_counts_image,
        se_image, composition_elements, weight_fractions, atomic_percentages,
    )


def get_1D_linescan(filepath, h5path, kp):
    """Read LineScans named in `kp` under the area at `h5path` in `filepath`.

    Return `(elements, element_maps, total_spectrum, dataset, metadata,
    total_counts_image, se_image, composition_elements, weight_fractions,
    atomic_percentages)` in measurement order. Each scan has parallel ROI
    lists of element symbols, profiles, and orbital labels in its host
    metadata dictionary, which also contains a copy of the element list.
    `total_spectrum` is the spectra list from
    `get_0D_spectrum`, and `dataset` contains the raw LSD arrays.
    Both image outputs contain one row per scan, with shape `(1, N)`.
    `total_counts_image` sums each LSD array along its last axis. `se_image`
    contains the field-of-view pixel values at `NPoints` evenly spaced
    positions from the stored start to end, using the pixel that contains
    each position. The final three lists report SPC composition in SPC
    element order.
    """
    kp = list(kp)
    elements = []
    element_maps = []
    dataset = []
    metadata = []
    total_counts_image = []
    se_image = []
    if kp:
        img, _ = get_image(filepath, h5path)
        height, width = img.shape[:2]
    with h5py.File(filepath, "r") as h5file:
        area = h5file[h5path]
        for name in kp:
            linescan = area[name]
            rois = linescan["ROIs"]
            roi_names = list(rois)
            roi_labels = [roi_name.split() for roi_name in roi_names]
            elements.append([element for element, orbital in roi_labels])
            element_maps.append([rois[roi_name][()] for roi_name in roi_names])
            scan_data = linescan["LSD"][()]
            dataset.append(scan_data)
            total_counts_image.append(scan_data.sum(axis=-1).reshape(1, -1))

            region = linescan["REGION"][0]
            point_count = int(region["NPoints"])
            x = np.linspace(region["X1"], region["X2"], point_count) * width
            y = np.linspace(region["Y1"], region["Y2"], point_count) * height
            cols = np.clip(np.floor(x).astype(int), 0, width - 1)
            rows = np.clip(np.floor(y).astype(int), 0, height - 1)
            se_image.append(img[rows, cols][None, :])
            scan_metadata = _get_host_metadata(linescan)
            scan_metadata["elements"] = elements[-1].copy()
            scan_metadata["orbitals"] = [orbital for element, orbital in roi_labels]
            metadata.append(scan_metadata)
    total_spectrum, _, composition_elements, weight_fractions, atomic_percentages = get_0D_spectrum(
        filepath, h5path, kp
    )
    return (
        elements, element_maps, total_spectrum, dataset, metadata, total_counts_image,
        se_image, composition_elements, weight_fractions, atomic_percentages,
    )



def get_xray_lines_exspy(elements):
    """Return eXSpy line energies, family weights, and names by atomic number.

    Each value contains four parallel arrays, sorted by energy. Energies
    are in keV and names use eXSpy notation such as Ka and Lb1. The fourth
    array contains None because this database supplies no transition-level
    labels. Weights are relative within each shell family.
    """
    line_spec = {}
    for element in elements:
        properties = exs.material.elements[element]
        lines = {
            name: line for name, line in properties.Atomic_properties.Xray_lines.as_dictionary().items()
            if isinstance(line, dict) and "energy (keV)" in line
        }
        names = np.array(list(lines))
        energies = np.array([line["energy (keV)"] for line in lines.values()])
        weights = np.array([line["weight"] for line in lines.values()])
        order = np.argsort(energies)
        line_spec[properties.General_properties.Z] = [
            energies[order], weights[order], names[order],
            np.full(len(names), None, dtype=object),
        ]
    return line_spec


def plot_xray_lines_exspy(ax, elements, linefactor=0.5, mpl_colorlist=None,
                           energie_range=None, dashed=True, linewidth=0.75,
                           legend_outside=True):
    """Overlay characteristic X-ray lines and add a separate element legend.

    Draw lines within the current x-limits and the optional exclusive
    `energie_range` in keV. Scale solid lines relative to the strongest
    visible line, whose height is `linefactor` of the plotting window.
    Heights use axes coordinates on both linear and logarithmic y-scales.
    Set `dashed` to add full-height guides. Colors cycle through
    `mpl_colorlist` or the Matplotlib defaults. Return `ax`.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    if not 0 <= linefactor <= 1:
        raise ValueError("linefactor must be between 0 and 1.")

    lowest, highest = sorted(ax.get_xlim())
    if energie_range is not None:
        lowest = max(lowest, energie_range[0])
        highest = min(highest, energie_range[1])

    elements = list(elements)
    line_spec = get_xray_lines_exspy(elements)
    visible_lines = []
    for element, color in zip(elements, cycle(_plot_colors(mpl_colorlist))):
        energies, intensities, *_ = line_spec[exs.material.elements[element].General_properties.Z]
        visible = (lowest < energies) & (energies < highest)
        if visible.any():
            visible_lines.append((element, color, energies[visible], intensities[visible]))
    if not visible_lines:
        return ax

    strongest = max(intensities.max() for _, _, _, intensities in visible_lines)
    handles = []
    for element, color, energies, intensities in visible_lines:
        heights = linefactor * intensities / strongest if strongest else np.zeros_like(intensities)
        for index, (energy, height) in enumerate(zip(energies, heights)):
            if dashed:
                ax.axvline(energy, color=color, linestyle="--", linewidth=linewidth,
                           alpha=0.35, zorder=1)
            line = ax.axvline(energy, ymax=height, color=color, linewidth=linewidth)
            if index == 0:
                line.set_label(element)
                handles.append(line)

    _place_plot_legends(ax, legend_outside, element_handles=handles)
    return ax

# Compatibility names for callers written before the eXSpy backend change.
get_xray_lines_xraydb = get_xray_lines_exspy
plot_xray_lines_xraydb = plot_xray_lines_exspy


def create_all_figures(experiment, output_path, element_list=None, logscale_y_minimum=0.5):
    """Save the figures for each spatial entry as PNGs in `output_path`.

    Save linear and logarithmic spectra for all spatial entries, field
    images for all entries, and maps for 1D and 2D. Set the 0D logarithmic
    lower limit in counts with `logscale_y_minimum`.
    Maps and map spectra use the first measurement in their entry. Annotate
    map spectra with its mapped elements. For 0D spectra, annotate the unique
    measured elements or the supplied `element_list`.
    """
    if not np.isfinite(logscale_y_minimum) or logscale_y_minimum <= 0:
        raise ValueError("logscale_y_minimum must be finite and positive.")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if element_list is not None:
        element_list = list(element_list)
    map_spectra = (("spectrum_linear", plot_spectrum_from_map),
                   ("spectrum_log", plot_spectrum_from_map))
    plots = {
        "0D": (("image", plot_0D_image), ("spectrum_linear", plot_0D_spectra),
               ("spectrum_log", plot_0D_spectra)),
        "1D": (("image", plot_1D_image), ("maps", plot_1D_maps), *map_spectra),
        "2D": (("image", plot_2D_image), ("maps", plot_2D_maps), *map_spectra),
    }
    for area, entry in experiment.items():
        dimension = area.rsplit(" spatial ", 1)[-1]
        for suffix, plot in plots.get(dimension, ()):
            options = {}
            if suffix == "spectrum_log":
                options = ({"logscale_y_minimum": logscale_y_minimum} if dimension == "0D"
                           else {"logscale_y": True})
            result = plot(experiment, area, **options)
            fig = result if dimension == "2D" and suffix == "maps" else result.figure
            try:
                if suffix.startswith("spectrum_"):
                    if dimension != "0D":
                        elements = list(dict.fromkeys(entry["elements"][0]))
                    elif element_list is None:
                        elements = sorted({element for group in entry["composition_elements"] for element in group})
                    else:
                        elements = element_list
                    plot_xray_lines_exspy(result, elements)
                fig.savefig(output_dir / f"{area} {suffix}.png")
            finally:
                plt.close(fig)


def get_background(spectrum,elements,peak_width=0.25,noise_level=150,plotting=True,
                   legend_outside=True):
    """Estimate the background of a two-row energy and counts spectrum.

    Exclude channels within `peak_width` keV of eXSpy lines for the supplied
    element symbols, interpolate across the excluded channels, and fit a
    smoothing spline controlled by `noise_level`. Return the fitted counts
    at the original energies. If `plotting` is true, plot the input,
    residual, and fitted background on logarithmic intensity axes.
    Set legend_outside=False to place legends inside the plotting rectangle.
    """
    energies=spectrum[0]
    spec=spectrum[1]
    m=len(energies)
    mask=np.ones(m,dtype=bool)
    line_spec=get_xray_lines_exspy(elements)
    #count=0
    all_energies=[]
    for element in line_spec:
        for index,energy in enumerate(line_spec[element][0]):
            all_energies.append(energy)
    for i in range(len(energies)):
        for j in all_energies:
            if np.abs(energies[i]-j)<peak_width:
                mask[i]=False
                
    background=np.interp(energies,energies[mask],spec[mask])
    spl=UnivariateSpline(energies,background)
    spl.set_smoothing_factor((m + np.sqrt(2*m)) * noise_level**2)
    if plotting:
        plt.semilogy(energies,spec,c='k',label='original data')
        plt.semilogy(energies,spec-spl(energies),'.',c='b',markersize=0.75,label='peaks removed data')
        plt.semilogy(energies,spl(energies),c='r',label='background')
        _place_plot_legends(plt.gca(), legend_outside)
        plt.xlabel('photon energy [keV]')
    return spl(energies)

def subtract_background(spectrum,background,lowermedianclip=0.5):
    """Subtract a channel-aligned background from an energy and counts spectrum.

    Shift the resulting counts so their minimum is zero, then clip them
    below `lowermedianclip` times the median of the shifted counts.
    Return a two-row array containing the original energies and the
    corrected counts.
    """
    energies=spectrum[0]
    spec=spectrum[1]
    bg_corr=np.zeros(spectrum.shape)
    bg_corr[0]=energies
    bg_corrected=spec-background
    bg_corrected -= np.min(bg_corrected)
    med=np.median(bg_corrected)
    bg_corr[1]=np.clip(bg_corrected,lowermedianclip*med,np.max(bg_corrected))
    return bg_corr
