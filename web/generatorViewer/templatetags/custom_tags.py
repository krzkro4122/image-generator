from django import template

register = template.Library()

@register.inclusion_tag('images_form.html')
def render_images_form(images):
    return {'images': images}

@register.inclusion_tag('image_form.html')
def render_image_form(image):
    return {'image': image}
