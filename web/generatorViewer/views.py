from django.http import Http404, HttpRequest, HttpResponse
from django.template import loader

from generatorViewer.src.views_utils import (
    load_generated_images,
    create_images,
    Keys,
)


def index(request: HttpRequest):
    if request.method != "GET":
        raise Http404("Method not allowed")

    batch_size = 8  # TODO - Get from UI in the request
    batch_index = 0  # TODO - Get from a DB or local storage

    image_tensors = load_generated_images(batch_size=batch_size, model_name="celeba")

    return HttpResponse(
        loader.get_template("images.html").render(
            request=request,
            context={
                Keys.IMAGES: create_images(
                    batch_index=batch_index,
                    batch_size=batch_size,
                    image_tensors=image_tensors,
                ),
            },
        )
    )


def next_batch(request: HttpRequest):
    if request.method != "POST":
        raise Http404("Method not allowed")

    request_batch_size = int(request.POST[Keys.BATCH_SIZE])

    image_tensors = load_generated_images(
        batch_size=request_batch_size, model_name="celeba"
    )

    return HttpResponse(
        loader.get_template("images_form.html").render(
            request=request,
            context={
                Keys.IMAGES: create_images(
                    batch_size=request_batch_size,
                    image_tensors=image_tensors,
                ),
            },
        )
    )
